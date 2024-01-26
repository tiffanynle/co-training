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

from itertools import zip_longest

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


class CoTrainingModel(torch.nn.Module):
    def __init__(self, models) -> None:
        super().__init__()
        self.models = models
        self.logs = []

    def forward(self, x) -> float:
        pass

    # some models may end training earlier than others,
    # but we would like to have training logged on the same step
    def _merge_wandb_logs(self, iteration, epochs, wandb_logs):
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
        assert (len(train_views) == len(val_views) == len(self.models),
                "number of views and number of models must be the same -- \
                got: train {}, val {}, models {}".format(
                    len(train_views), len(val_views), len(self.models)))

        optimizers = []
        schedulers = []
        stoppers = []
        samplers_train = []

        for model in self.models:
            optimizers.append(optimizer(model.parameters(),
                                        **optimizer_kwargs))
            stoppers.append(EarlyStopper(**stopper_kwargs))

        if lr_scheduler is not None:
            for opt in optimizers:
                schedulers.append(lr_scheduler(opt, 'max'))

        states = {f'model{i}_state': model.state_dict() 
                  for i, model in enumerate(self.models)}
        states.update({f'optimizer{i}_state': opt.state_dict() 
                       for i, opt in enumerate(optimizers)})

        # update views to contain dataloaders instead of datasets
        for i in range(len(train_views)):
            sampler_train = DistributedSampler(train_views[i], rank=rank,
                                               num_replicas=world_size)
            samplers_train.append(sampler_train)

            cuda_kwargs = {'num_workers': 12, 'pin_memory': True, 'shuffle': False}
            loader_kwargs = {'batch_size': batch_size, 'sampler': sampler_train}
            loader_kwargs.update(**cuda_kwargs)

            loader = DataLoader(train_views[i], **loader_kwargs)
            train_views[i] = loader
            
            sampler_val = DistributedSampler(val_views[i], rank=rank, 
                                             num_replicas=world_size)

            loader_kwargs.update({'sampler': sampler_val})
            loader = DataLoader(val_views[i], **loader_kwargs)
            val_views[i] = loader

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
                                               model, train_views[i], 
                                               loss_fn, optimizers[i])
                val_acc, val_loss = test_ddp(rank, device, model, 
                                         val_views[i], loss_fn)
                
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

        self.logs += self._merge_wandb_logs(iteration, epochs, iteration_logs)
    
        return states, best_val_acc