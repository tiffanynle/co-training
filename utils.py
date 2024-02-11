import torch
from torch import nn
from torch.nn import functional as F

from torch.multiprocessing import set_start_method, set_forkserver_preload
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import numpy as np


class EarlyStopper:
    def __init__(self,
                 metric='accuracy',
                 patience=0,
                 min_delta=0.0):
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta

        self.early_stop = False
        self.epochs_since_improvement = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

    def step(self, val_acc, val_loss):
        self.epochs_since_improvement += 1
        if self.metric == 'loss' and val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.epochs_since_improvement = 0
        elif self.metric == 'accuracy' and val_acc > self.best_val_acc + self.min_delta:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.epochs_since_improvement = 0
        if self.epochs_since_improvement > self.patience:
            self.early_stop = True


def create_imagefolder(data, samples, path, transform, new_path=None):
    imgfolder = datasets.ImageFolder(path, transform=transform)
    imgfolder.class_to_idx = data['class_map']
    imgfolder.classes = list(data['class_map'].keys())
    imgfolder.samples = samples

    if new_path is not None:
        imgfolder.root = new_path

    return imgfolder


def create_sampler_loader(rank, world_size, 
                          data,
                          batch_size=64,
                          cuda_kwargs={'num_workers': 12, 
                                       'pin_memory': True, 
                                       'shuffle': False}, 
                                       shuffle=True,
                                       persistent_workers=False):
    sampler = DistributedSampler(data, rank=rank, num_replicas=world_size, shuffle=shuffle)

    loader_kwargs = {'batch_size': batch_size, 
                     'sampler': sampler, 
                     'multiprocessing_context': 'forkserver', 
                     'persistent_workers': persistent_workers}
    loader_kwargs.update(cuda_kwargs)

    loader = DataLoader(data, **loader_kwargs)

    return sampler, loader


def add_to_imagefolder(paths, labels, dataset):
    """
    Adds the paths with the labels to an image classification dataset

    :list paths: a list of absolute image paths to add to the dataset
    :list labels: a list of labels for each path
    :Dataset dataset: the dataset to add the samples to
    """

    new_samples = list(zip(paths, labels))

    dataset.samples += new_samples

    return dataset


def setup(rank, world_size):
    set_start_method('forkserver')
    set_forkserver_preload(['torch'])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def accuracy(output, target, topk=(1,)):
    output = output.to(torch.device('cpu'))
    target = target.to(torch.device('cpu'))
    maxk = max(topk)
    batch_size = target.shape[0]

    _, idx = output.sort(dim=1, descending=True)
    pred = idx.narrow(1, 0, maxk).t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(dim=0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def epoch_accuracy(loader_s, loader_t, student, teacher):
    out_ct = [((student(L_s.to(0)), y_s), (teacher(L_t.to(0)), y_t)) for (L_s, y_s), (L_t, y_t) in zip(iter(loader_s), iter(loader_t))]

    out_epoch_s = [accuracy(a[0], a[1])[0].cpu().item() for a, _ in out_ct]
    out_epoch_t = [accuracy(b[0], b[1])[0].cpu().item() for _, b in out_ct]
    out_epoch_ct = [accuracy(torch.nn.Softmax(dim=-1)(a[0])*torch.nn.Softmax(dim=-1)(b[0]), a[1])[0].cpu().item() for a, b in out_ct]
    
    return sum(out_epoch_s) / len(out_epoch_s), sum(out_epoch_t) / len(out_epoch_t), sum(out_epoch_ct) / len(out_epoch_t)
