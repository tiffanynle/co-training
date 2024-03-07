from itertools import zip_longest

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import *


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
    labels = []
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

            # same as above but we need the labels
            label_list = [torch.full((batch_size,), -1, 
                                     dtype=torch.float16).to(device)
                                     for _ in range(world_size)]

            pad2 = torch.full((batch_size,), -1, 
                               dtype=torch.float16).to(device)
            pad2[:y.shape[0]] = y

            # all-gather the full list of predictions across all processes
            dist.all_gather(tensor_list, pad)
            batch_outputs = torch.cat(tensor_list)

            # all-gather the full list of labels
            dist.all_gather(label_list, pad2)
            batch_labels = torch.cat(label_list)

            # remove all rows of the tensor that contain a -1
            # (as this is not a valid value anywhere)
            mask = ~(batch_outputs == -1).any(-1)
            batch_outputs = batch_outputs[mask]
            predictions.append(batch_outputs)

            batch_labels = batch_labels[mask]
            labels.append(batch_labels)

    return torch.cat(predictions), torch.cat(labels)


# def co_test_ddp(rank, device, models, loaders):
#     for model in models:
#         model.eval()

#     iters_loaders = [iter(loader) for loader in loaders]
#     ddp_loss = torch.zeros(3).to(device)
#     with torch.no_grad():
#         pass
#     dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
#     test_acc = ddp_loss[1] / ddp_loss[2]

#     if rank == 0:
#         print('Test error: \tCo-Accuracy: {:.2f}%'
#               .format(100*test_acc))
    
#     return test_acc

def c_test(rank, model0, model1, loader0, loader1, device):
    ddp_loss = torch.zeros(3).to(device)
    model0.eval()
    model1.eval()
    with torch.no_grad():
        for batch, ((X0, y0), (X1, y1))  in enumerate(zip(iter(loader0), iter(loader1))):
            X0, y0 = X0.to(device), y0.to(device)
            X1, y1 = X1.to(device), y1.to(device)

            output = torch.nn.Softmax(-1)(model0(X0)) * torch.nn.Softmax(-1)(model1(X1))

            ddp_loss[1] += (output.argmax(1) == y1).type(torch.float).sum().item()
            ddp_loss[2] += len(X0)
    
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    test_acc = ddp_loss[1] / ddp_loss[2] 
    test_loss = ddp_loss[0] / ddp_loss[2]

    if rank == 0:
        print('Test error: \tCo-Accuracy: {:.2f}%'
              .format(100*test_acc))

    return test_acc


def get_topk(prediction, k, frequencies):
    prob, label = torch.max(prediction, 1)

    unique, counts = np.unique(label.cpu().numpy(), return_counts=True)

    counts = frequencies

    count_per_class = k * (np.ones_like(frequencies) / frequencies.shape[0])
    # ok, but this is not exactly n% we will have some rounding to do here
    idx = []
    while len(idx) < k:
        idx = torch.cat([torch.argsort(prob[torch.where(label == l)], descending=True)[:cascade_round(count_per_class)[l]] for l in unique])
        count_per_class += (1 / frequencies.shape[0])

    idx = idx[:k]
    # assert len(idx) == k # ...?

    unique, counts = np.unique(label[idx].cpu().numpy(), return_counts=True)
    print(counts, frequencies)

    return prob[idx], label[idx], idx


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


class CoTrainingModel:
    def __init__(self, 
                 rank: int, 
                 world_size: int, 
                 models: list) -> None:
        self.rank = rank
        self.world_size = world_size
        self.models = models
        self.logs = []
        self.frequencies = np.array([])

    def predict(self, device, unlbl_views, num_classes, batch_size):
        samplers_unlbl = []
        loaders_unlbl = []
        for i in range(len(unlbl_views)):
            sampler_unlbl, loader_unlbl = create_sampler_loader(self.rank, 
                                                                self.world_size,
                                                                unlbl_views[i],
                                                                batch_size,
                                                                shuffle=False)
            samplers_unlbl.append(sampler_unlbl)
            loaders_unlbl.append(loader_unlbl)

        if self.rank == 0:
            print("making predictions on unlabeled set...")

        # make predictions for everything in unlabeled set, for all models
        preds_softmax = []
        for i, model in enumerate(self.models):
            preds, labels = predict_ddp(self.world_size, device, 
                               model, loaders_unlbl[i], 
                               batch_size, 
                               num_classes)
            preds = preds[:len(unlbl_views[i])]
            labels = labels[:(len(unlbl_views[i]))]
            # ensure all processes agree on the list of predictions
            dist.broadcast(preds, 0)
            preds_softmax.append(preds)
        dist.broadcast(labels, 0)

        if self.rank == 0:
            print(f"shape of predictions {preds.shape} of labels {labels.shape}")

        # num views x num instances x num classes
        preds_softmax = torch.stack(preds_softmax)

        return preds_softmax, labels

    def update(self,
               preds_softmax: torch.Tensor,
               train_views: list, 
               unlbl_views: list,
               k_total: float) -> None:
        msg = (f'number of views and number of models must be the same -- '
               f'got train: {len(train_views)}, unlabeled: {len(unlbl_views)}, '
               f'models: {len(self.models)}')
        assert len(train_views) == len(unlbl_views) == len(self.models), msg

        # assuming that k is # of total dataset to bring in
        # we'll have each model take in top-(k // len(models)) predictions
        # (though we may bring in less than this.) (?)
        k_model = k_total // len(self.models)

        if self.rank == 0:
            print("updating datasets...")

        # if only 2 views, we'll filter out conflicting predictions
        # and then update the datasets by swapping labels
        if len(self.models) == 2:
            lbls_topk = []
            idxes_topk = []
            for pred in preds_softmax:
                _, lbl_topk, idx_topk = get_topk(pred,
                                                 k_model if k_model <= len(pred)
                                                 else len(pred), self.frequencies)
                lbls_topk.append(lbl_topk.detach().cpu().numpy())
                idxes_topk.append(idx_topk.detach().cpu().numpy())

            self._resolve_conflicts_twoview(lbls_topk, idxes_topk)
            self._update_datasets_twoview(lbls_topk, idxes_topk,
                                         train_views, unlbl_views)
        # if views > 2, aggregate the predictions and call it a day
        else:
            self._update_datasets_multiview(preds_softmax, k_model, 
                                            train_views, unlbl_views)

    # 2 views, no ensembling
    def _resolve_conflicts_twoview(self, lbls_topk, idxes_topk):
        msg = (f'expected predictions for 2 views, got: '
               f'labels {len(lbls_topk)}, indices {len(idxes_topk)}')
        assert len(lbls_topk) == len(idxes_topk) == 2, msg

        inter, idx_inter0, idx_inter1 = np.intersect1d(idxes_topk[0],
                                                       idxes_topk[1],
                                                       return_indices=True)
        
        mask_colls = lbls_topk[0][idx_inter0] != lbls_topk[1][idx_inter1]
        idx_colls = inter[mask_colls]

        if len(idx_colls) > 0:
            if self.rank == 0:
                print(f"number of conflicting predictions: {len(idx_colls)}")
            idx_coll0 = idx_inter0[mask_colls]
            idx_coll1 = idx_inter1[mask_colls]
            
            mask = np.ones(len(idxes_topk[0]), dtype=bool)
            mask[idx_coll0] = False
            mask[idx_coll1] = False

            for i in range(len(lbls_topk)):
                lbls_topk[i] = lbls_topk[i][mask]
                idxes_topk[i] = idxes_topk[i][mask]

    def _update_datasets_twoview(self, lbls_topk, idxes_topk,
                                train_views, unlbl_views):
        samples_unlbl0 = np.stack([np.array(a) 
                                   for a in unlbl_views[0].samples])
        samples_unlbl1 = np.stack([np.array(a) 
                                   for a in unlbl_views[1].samples])

        # retrieve the instances that have been labeled 
        # with high confidence by the other model
        list_samples0 = [(str(a[0]), int(a[1])) 
                         for a in list(samples_unlbl0[idxes_topk[1]])]
        list_samples1 = [(str(a[0]), int(a[1])) 
                         for a in list(samples_unlbl1[idxes_topk[0]])]
        
        # image paths for both
        paths0 = [i for i, _ in list_samples0]
        paths1 = [i for i, _ in list_samples1]

        # update imagefolders
        train_views[0] = add_to_imagefolder(paths0, lbls_topk[1].tolist(), 
                                            train_views[0])
        train_views[1] = add_to_imagefolder(paths1, lbls_topk[0].tolist(), 
                                            train_views[1])

        # remove instances from unlabeled dataset
        mask = np.ones(len(unlbl_views[0]), dtype=bool)
        for idx_topk_i in idxes_topk:
            mask[idx_topk_i] = False
        
        if self.rank == 0:
            print(f"number of unlabeled instances to remove: {(~mask).sum()}")

        samples_unlbl0 = samples_unlbl0[mask]
        samples_unlbl1 = samples_unlbl1[mask]

        list_unlbl0 = [(str(a[0]), int(a[1])) for a in list(samples_unlbl0)]
        list_unlbl1 = [(str(a[0]), int(a[1])) for a in list(samples_unlbl1)]

        unlbl_views[0].samples = list_unlbl0
        unlbl_views[1].samples = list_unlbl1

        if self.rank == 0:
            print(f"remaining number of unlabeled instances: {len(unlbl_views[0])}")

    def _update_datasets_multiview(self, k_model, preds_softmax,
                                  train_views, unlbl_views):
        num_views = len(self.models)
        num_instances = preds_softmax.shape[1]
        mask = np.ones(num_instances, dtype=bool)

        for i in range(num_views):
            samples_unlbl = np.stack([np.array(a)
                                      for a in unlbl_views[i].samples])
            
            # mask out softmax predictions to exclude "student" predictions
            mask_preds = torch.ones(num_views, dtype=torch.bool)
            mask_preds[i] = False
            preds_agg = torch.prod(preds_softmax[mask_preds], dim=0) / num_views
            _, lbl_topk, idx_topk = get_topk(preds_agg, k_model 
                                             if k_model <= num_instances
                                             else num_instances, self.frequencies)

            # training set update
            list_samples = [str(a[0], int(a[1]))
                            for a in list(samples_unlbl[idx_topk])]
            paths = [i for i, _ in list_samples]
            train_views[i] = add_to_imagefolder(paths, lbl_topk.tolist(), 
                                                train_views[i])

            # update mask to remove now-pseudolabeled instances (later)
            mask[idx_topk] = False

        # remove pseudolabeled instances from unlabeled set
        for i in range(num_views):
            samples_unlbl = np.stack([np.array(a) 
                                      for a in unlbl_views[i].samples])
            samples_unlbl = samples_unlbl[mask]
            list_unlbl = [(str(a[0]), int(a[1])) for a in list(samples_unlbl)]
            unlbl_views[i].samples = list_unlbl
        
        if self.rank == 0:
            print(f"number of unlabeled instances to remove: {(~mask).sum()}")
            print(f"remaining number of unlabeled instances: {len(unlbl_views[0])}")

    # one full pass of co-training without dataset updates
    def train(self,
              device: torch.device,
              iteration: int, 
              epochs: int,
              train_views: list,
              val_views: list,
              test_views: list,
              batch_size: int = 64,
              optimizer: torch.optim.Optimizer = SGD,
              optimizer_kwargs: dict = {'lr': 1e-3,
                                        'momentum': 0.9},
              stopper_kwargs: dict = {'metric': 'accuracy',
                                      'patience': 32,
                                      'min_delta': 1e-3},
              lr_scheduler: torch.optim.lr_scheduler = ReduceLROnPlateau) -> None:
        msg = (f'number of views and number of models must be the same -- '
               f'got train: {len(train_views)}, val: {len(val_views)}, '
               f'test: {len(test_views)}, models: {len(self.models)}')
        assert len(train_views) == len(val_views) == len(test_views) == len(self.models), msg

        optimizers = []
        schedulers = []
        stoppers = []

        states = dict()
        for i in range(len(self.models)):
            stoppers.append(EarlyStopper(**stopper_kwargs))
            optimizers.append(optimizer(self.models[i].parameters(), 
                                        **optimizer_kwargs))
            states[f'model{i}_state'] = self.models[i].state_dict()
            states[f'optimizer{i}_state'] = optimizers[i].state_dict()

        if lr_scheduler is not None:
            for i, opt in enumerate(optimizers):
                schedulers.append(lr_scheduler(opt)) 

        samplers_train, loaders_train = create_samplers_loaders(self.rank, self.world_size,
                                                                train_views, batch_size, 
                                                                persistent_workers=True)
        
        samplers_val, loaders_val = create_samplers_loaders(self.rank, self.world_size,
                                                            val_views, batch_size, 
                                                            persistent_workers=True)
        
        samplers_test, loaders_test = create_samplers_loaders(self.rank, self.world_size,
                                                              test_views, batch_size, 
                                                              persistent_workers=True)

        weight = (-1*torch.log(torch.tensor(self.frequencies))).type(torch.float)
        weight = weight.to(device)
        loss_fn = nn.CrossEntropyLoss(weight=weight)

        best_val_acc = 0.0
        best_val_loss = float('inf')
        iteration_logs = []
        for i in range(len(self.models)):
            if self.rank == 0:
                print(f"training view{i}...")
            step = 0 * iteration
            model_logs = []
            for epoch in range(epochs):
                step += 1
                samplers_train[i].set_epoch(epoch)
                train_acc, train_loss = train_ddp(self.rank, device, epoch, 
                                               self.models[i], loaders_train[i], 
                                               loss_fn, optimizers[i])
                val_acc, val_loss = test_ddp(self.rank, device, self.models[i], 
                                         loaders_val[i], loss_fn)
                test_acc, test_loss = test_ddp(self.rank, device, self.models[i],
                                               loaders_test[i], loss_fn)

                stoppers[i].step(val_acc, val_loss)
                if stoppers[i].epochs_since_improvement == 0:
                    states[f'model{i}_state'] = self.models[i].state_dict()
                    states[f'optimizer{i}_state'] = optimizers[i].state_dict()
                    best_val_acc = max(best_val_acc, stoppers[i].best_val_acc)
                    best_val_loss = min(best_val_loss, stoppers[i].best_val_loss)
                
                model_logs.append({f'train_acc{i}': train_acc,
                                  f'train_loss{i}': train_loss,
                                  f'val_acc{i}': val_acc,
                                  f'val_loss{i}': val_loss,
                                  f'test_acc{i}': test_acc,
                                  f'test_loss{i}': test_loss,
                                  f'best_val_acc{i}': stoppers[i].best_val_acc,
                                  f'best_val_loss{i}': stoppers[i].best_val_loss
                                  })

                if stoppers[i].early_stop:
                    break
                
                schedulers[i].step(val_loss)

            # need to manually shut down the workers as they will persist otherwise
            loaders_train[i]._iterator._shutdown_workers()
            loaders_val[i]._iterator._shutdown_workers()
            loaders_test[i]._iterator._shutdown_workers()

            iteration_logs.append(model_logs)

        self.logs += merge_wandb_logs(iteration, epochs, iteration_logs)
    
        return best_val_acc, best_val_loss, states
