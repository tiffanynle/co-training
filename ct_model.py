import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, DistributedSampler

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import wandb
import numpy as np

from itertools import zip_longest, reduce

from utils import *
from ct_utils import *


def train_ddp(rank, device, epoch, model, loader, loss_fn, optimizer):
    ddp_loss = torch.zeros(3).to(device)
    model.train()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(X)
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += (output.argmax(1) == y).type(torch.float).sum().item()
        ddp_loss[2] += len(X)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    train_acc = ddp_loss[1] / ddp_loss[2] 
    train_loss = ddp_loss[0] / ddp_loss[2]

    if rank == 0:
        print('Train Epoch: {} \tAccuracy: {:.2f}% \tAverage Loss: {:.6f}'
              .format(epoch, 
                      100*(ddp_loss[1] / ddp_loss[2]), 
                      ddp_loss[0] / ddp_loss[2]))

    return train_acc, train_loss 


def test_ddp(rank, device, model, loader, loss_fn):
    ddp_loss = torch.zeros(3).to(device)
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = loss_fn(output, y)
            ddp_loss[0] += loss.item()
            ddp_loss[1] += (output.argmax(1) == y).type(torch.float).sum().item()
            ddp_loss[2] += len(X)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    test_acc = ddp_loss[1] / ddp_loss[2] 
    test_loss = ddp_loss[0] / ddp_loss[2]

    if rank == 0:
        print('Test error: \tAccuracy: {:.2f}% \tAverage loss: {:.6f}'
              .format(100*test_acc, test_loss))

    return test_acc, test_loss


def predict_ddp(world_size, device, model, loader, batch_size, num_classes):
    model.eval()
    predictions = []
    softmax = torch.nn.Softmax(-1)
    with torch.no_grad():
        for X, y in loader:
            tensor_list = [torch.full((batch_size, num_classes), -1,
                                      dtype=torch.float16).to(device) 
                           for _ in range(world_size)]
            output = softmax(model(X.to(device)))

            # pad the output so that it contains the same 
            # number of rows as specified by the batch size
            pad = torch.full((batch_size, num_classes), -1, 
                             dtype=torch.float16).to(device)
            pad[:output.shape[0]] = output

            # all-gather the full list of predictions across all processes
            dist.all_gather(tensor_list, pad)
            batch_outputs = torch.cat(tensor_list)

            # remove all rows of the tensor that contain a -1
            # (as this is not a valid value anywhere)
            mask = ~(batch_outputs == -1).any(-1)
            batch_outputs = batch_outputs[mask]
            predictions.append(batch_outputs)

    return torch.cat(predictions)


def get_topk(prediction, k):
    prob, label = torch.max(prediction, 1)
    idx = torch.argsort(prob, descending=True)[:k]
    return label[idx], idx


# some models may end training earlier than others,
# but we would like to have training logged on the same step
def merge_wandb_logs(iteration, epochs, wandb_logs) -> list:
    logs_with_steps = []
    padded_logs = list(zip_longest(*wandb_logs, fillvalue=None))
    for epoch, logs in enumerate(padded_logs):
        log_merged = {}
        for log in logs:
            if log is None: log = {}
            log_merged = log_merged | log
        step = epoch + iteration*epochs
        logs_with_steps.append((log_merged, step))
    return logs_with_steps


class CoTrainingModel(torch.nn.Module):
    def __init__(self, models) -> None:
        super().__init__()
        self.models = models
        self.logs = []

    def forward(self, x) -> float:
        pass

    # update labeled and unlabeled datasets
    def update(self,
               rank, world_size, device,
               train_views, unlbl_views,
               num_classes, k_total,
               batch_size=64) -> None:
        assert (len(train_views) == len(unlbl_views) == len(self.models),
                "number of views and number of models must be the same -- \
                got: train {}, unlabeled {}, models {}".format(
                    len(train_views), len(unlbl_views), len(self.models)))
        
        samplers_unlbl = []
        loaders_unlbl = []
        for i in range(len(unlbl_views)):
            sampler_unlbl = DistributedSampler(unlbl_views[i], rank=rank,
                                               num_replicas=world_size)
            samplers_unlbl.append(sampler_unlbl)

            cuda_kwargs = {'num_workers': 12, 
                           'pin_memory': True, 
                           'shuffle': False}
            loader_kwargs = {'batch_size': batch_size, 
                             'sampler': sampler_unlbl}
            loader_kwargs.update(**cuda_kwargs)

            loader = DataLoader(unlbl_views[i], **loader_kwargs)
            loaders_unlbl.append(loader)

        # make predictions for everything in unlabeled set, for all models
        preds_softmax = []
        for i, model in enumerate(self.models):
            pred = predict_ddp(world_size, device, 
                               model, loaders_unlbl[i], 
                               batch_size, 
                               num_classes)[:len(unlbl_views[i])]
            # ensure all processes agree on the list of predictions
            dist.broadcast(pred, 0)
            preds_softmax.append(pred)

        # assuming that k is % of total dataset to bring in
        # we'll have each model take in top-(k // len(models)) predictions
        # (though we may bring in less than this if there are conflicts)
        k_model = k_total // len(self.models)

        lbls_topk = []
        idxes_topk = []
        # surely there is a better way to write this...
        for pred in preds_softmax:
            lbl_topk, idx_topk = get_topk(pred,
                                          k_model if k_model <= len(pred)
                                          else len(pred))
            lbls_topk.append(lbl_topk)
            idxes_topk.append(idx_topk)

        # if only 2 views, we'll filter out conflicting predictions
        # and then update the datasets by swapping labels
        if len(self.models) == 2:
            lbls_topk, idxes_topk = self.resolve_conflicts_twoview(lbls_topk, 
                                                                   idxes_topk)
            self.update_datasets_twoview(lbls_topk, idxes_topk,
                                         train_views, unlbl_views)
        # if views > 2, we'll have to ensemble predictions :(
        else:
            pass

    # 2 views, no ensembling
    def resolve_conflicts_twoview(self, lbls_topk, idxes_topk):
        assert (len(lbls_topk) == len(idxes_topk) == 2), \
                "expected predictions for 2 views, got: labels {}, indexes {}" \
                .format(len(lbls_topk), len(idxes_topk))
        lbls_np = []
        idxes_np = []
        for i in range(len(lbls_topk)):
            lbl_np = lbls_topk[i].detach().cpu().numpy()
            idx_np = idxes_topk[i].detach().cpu().numpy()
            lbls_np.append(lbl_np)
            idxes_np.append(idx_np)

        inter, idx_inter0, idx_inter1 = np.intersect1d(idxes_np[0],
                                                       idxes_np[1],
                                                       return_indices=True)
        
        mask_colls = lbls_np[0][idx_inter0] != lbls_np[1][idx_inter1]
        idx_colls = inter[mask_colls]

        if len(idx_colls) > 0:
            print(f"number of conflicting predictions: {len(idx_colls)}")
            idx_coll0 = idx_inter0[mask_colls]
            idx_coll1 = idx_inter1[mask_colls]
            
            mask = np.ones(len(idxes_topk[0]), dtype=bool)
            mask[idx_coll0] = False
            mask[idx_coll1] = False

            for i in range(len(lbls_np)):
                lbls_np[i] = lbls_np[i][mask]
                idxes_np[i] = idxes_np[i][mask]
        
        return lbls_np, idxes_np

    def update_datasets_twoview(self, lbls_topk, idxes_topk,
                                train_views, unlbl_views):
        samples_unlbl0 = np.stack([np.array(a) 
                                   for a in unlbl_views[0].samples])
        samples_unlbl1 = np.stack([np.array(a) 
                                   for a in unlbl_views[1].samples])

        # retrieve the instances that have been labeled with high confidence by the other model
        list_samples0 = [(str(a[0]), int(a[1])) 
                         for a in list(samples_unlbl0[idxes_topk[1]])]
        list_samples1 = [(str(a[0]), int(a[1])) 
                         for a in list(samples_unlbl1[idxes_topk[0]])]
        
        # image paths for both
        paths0 = [i for i, _ in list_samples0]
        paths1 = [i for i, _ in list_samples1]

        # update imagefolders
        train_views[0].samples += list(zip(paths0, lbls_topk[1].tolist()))
        train_views[1].samples += list(zip(paths1, lbls_topk[0].tolist()))

        # remove instances from unlabeled dataset
        mask = np.ones(len(unlbl_views[0]), dtype=bool)
        for idx_topk_i in idxes_topk:
            mask[idx_topk_i] = False
        
        print(f"number of unlabeled instances to remove: {(~mask).sum()}")

        samples_unlbl0 = samples_unlbl0[mask]
        samples_unlbl1 = samples_unlbl1[mask]

        list_unlbl0 = [(str(a[0]), int(a[1])) for a in list(samples_unlbl0)]
        list_unlbl1 = [(str(a[0]), int(a[1])) for a in list(samples_unlbl1)]

        unlbl_views[0].samples = list_unlbl0
        unlbl_views[1].samples = list_unlbl1

        print(f"remaining number of unlabeled instances: {len(unlbl_views[0])}")

    # views > 2, ensembling :(
    def resolve_conflicts_multiview(self, lbls_topk, idxes_topk):
        assert (len(lbls_topk) == len(self.models) == len(idxes_topk), \
                "expected predictions for {} views, got: labels {}, indexes {}" \
                .format(len(self.models), len(lbls_topk), len(idxes_topk)))
        lbls_np = []
        for lbls in lbls_topk:
            lbl_np = lbls.detach().cpu().numpy()
            lbls_np.append(lbl_np)

        idxes_np = []
        for idxes in idxes_topk:
            idx_np = idxes.detach().cpu().numpy()
            idxes_np.append(idx_np) 

        # cannot fetch indices of intersection with a reduction
        # have to find another way to do this...
        # inter = reduce(np.intersect1d, lbls_np)

    def update_datasets_multiview(self, lbls_topk, idxes_topk,
                                  train_views, unlbl_views):
        pass

    # one full pass of co-training without dataset updates
    def train(self,
              rank, world_size, device,
              iteration, epochs,
              train_views, val_views, 
              batch_size=64,
              optimizer=SGD,
              optimizer_kwargs={'lr':1e-3,
                                'momentum': 0.9},
              stopper_kwargs={'metric': 'accuracy',
                              'patience': 32,
                              'min_delta': 1e-3,
                              'warmup': 0},
              lr_scheduler=ReduceLROnPlateau) -> None:
        assert len(train_views) == len(val_views) == len(self.models), \
                "number of views and number of models must be the same -- \
                got: train {}, val {}, models {}".format(
                    len(train_views), len(val_views), len(self.models))

        optimizers = []
        schedulers = []
        stoppers = []
        samplers_train = []
        samplers_val = []
        loaders_train = []
        loaders_val = []

        for model in self.models:
            optimizers.append(optimizer(model.parameters(),
                                        **optimizer_kwargs))
            stoppers.append(EarlyStopper(**stopper_kwargs))

        if lr_scheduler is not None:
            for opt in optimizers:
                schedulers.append(lr_scheduler(opt))

        states = {f'model{i}_state': model.state_dict() 
                  for i, model in enumerate(self.models)}
        states.update({f'optimizer{i}_state': opt.state_dict() 
                       for i, opt in enumerate(optimizers)})

        # update views to contain dataloaders instead of datasets
        for i in range(len(train_views)):
            sampler_train = DistributedSampler(train_views[i], rank=rank,
                                               num_replicas=world_size)
            samplers_train.append(sampler_train)

            cuda_kwargs = {'num_workers': 12, 
                           'pin_memory': True, 
                           'shuffle': False}
            loader_kwargs = {'batch_size': batch_size, 
                             'sampler': sampler_train}
            loader_kwargs.update(**cuda_kwargs)

            loader = DataLoader(train_views[i], **loader_kwargs)
            loaders_train.append(loader)
            
            sampler_val = DistributedSampler(val_views[i], rank=rank, 
                                             num_replicas=world_size)
            samplers_val.append(sampler_val)

            loader_kwargs.update({'sampler': sampler_val})
            loader = DataLoader(val_views[i], **loader_kwargs)
            loaders_val.append(loader)

        loss_fn = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        iteration_logs = []
        for i, model in enumerate(self.models):
            step = 0 * iteration
            model_logs = []
            for epoch in range(epochs):
                step += 1
                samplers_train[i].set_epoch(epoch)
                train_acc, train_loss = train_ddp(rank, device, epoch, 
                                               model, loaders_train[i], 
                                               loss_fn, optimizers[i])
                val_acc, val_loss = test_ddp(rank, device, model, 
                                         loaders_val[i], loss_fn)
                
                # if rank == 0:
                #     wandb.log({f'train_acc{i}': train_acc,
                #                f'train_loss{i}': train_loss,
                #                f'val_acc{i}': val_acc,
                #                f'val_loss{i}': val_loss},
                #                step=step)
                model_logs.append({f'train_acc{i}': train_acc,
                                  f'train_loss{i}': train_loss,
                                  f'val_acc{i}': val_acc,
                                  f'val_loss{i}': val_loss})
                    
                schedulers[i].step(val_acc)
                stoppers[i].step(epoch, val_acc, val_loss)
                
                if stoppers[i].counter == 0:
                    best_val_acc = max(best_val_acc, stoppers[i].best_val_acc)
                    states[f'model{i}_state'] = model.state_dict()
                    states[f'optimizer{i}_state'] = optimizers[i].state_dict()
                if stoppers[i].early_stop:
                    break

            iteration_logs.append(model_logs)

        self.logs += merge_wandb_logs(iteration, epochs, iteration_logs)
    
        return states, best_val_acc