import torch
from torch import nn
from torch.nn import functional as F

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

import numpy as np


class EarlyStopper:
    def __init__(self,
                 metric='accuracy',
                 patience=0,
                 min_delta=0.0,
                 warmup=0):
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup

        self.early_stop = False
        self.counter = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

    def step(self, epoch, val_acc, val_loss):
        if (epoch > self.warmup):
            self.counter += 1
            if self.metric == 'loss' and val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.counter = 0
            elif self.metric == 'accuracy' and val_acc > self.best_val_acc + self.min_delta:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.counter = 0
        if self.counter > self.patience:
            self.early_stop = True


def create_imagefolder(data, samples, path, transform, new_path=None):
    imgfolder = datasets.ImageFolder(path, transform=transform)
    imgfolder.class_to_idx = data['class_map']
    imgfolder.classes = list(data['class_map'].keys())
    imgfolder.samples = samples

    if new_path is not None:
        imgfolder.root = new_path

    return imgfolder


def create_sampler_loader(args, rank, world_size, data,
                           cuda_kwargs={'num_workers': 12, 
                                        'pin_memory': True, 
                                        'shuffle': False}, 
                                        shuffle=True):
    sampler = DistributedSampler(data, rank=rank, 
                                 num_replicas=world_size, shuffle=shuffle)

    loader_kwargs = {'batch_size': args.batch_size, 'sampler': sampler}
    loader_kwargs.update(cuda_kwargs)

    loader = DataLoader(data, **loader_kwargs)

    return sampler, loader


def get_topk_predictions(pred, k):
    prob, label = torch.max(pred, 1)
    idx = torch.argsort(prob, descending=True)[:k]
    return prob[idx], label[idx], idx

def predict(world_size, batch_size, loader, model, num_classes, device):
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

def setup(rank, world_size):
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