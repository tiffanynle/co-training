import random
from math import ceil, floor
from typing import Literal

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.multiprocessing import set_forkserver_preload, set_start_method
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


class EarlyStopper:
    def __init__(self,
                 metric: Literal['loss', 'accuracy'] = 'accuracy',
                 patience: int = 0,
                 min_delta: float = 0.0):
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta

        self.early_stop = False
        self.epochs_since_improvement = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0

    def step(self, val_acc: float, val_loss: float):
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


def create_sampler_loader(rank: int, 
                          world_size: int, 
                          data: torch.utils.data.Dataset,
                          batch_size: int = 64,
                          cuda_kwargs: dict = {'num_workers': 12, 
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


def create_samplers_loaders(rank: int, 
                            world_size: int, 
                            views: list[torch.utils.data.Dataset], 
                            batch_size: int = 64,
                            cuda_kwargs: dict = {'num_workers': 12, 
                                                 'pin_memory': True, 
                                                 'shuffle': False},
                            shuffle: bool = True,  
                            persistent_workers: bool = False):
    samplers = []
    loaders = []
    for i in range(len(views)):
        sampler, loader = create_sampler_loader(rank, world_size, views[i], 
                                                batch_size, cuda_kwargs, 
                                                shuffle, persistent_workers)
        samplers.append(sampler)
        loaders.append(loader)
    
    return samplers, loaders


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


# TODO write multi-view implementation (...)
def train_test_split_samples(samples0, samples1, test_size, random_state=None):
    if random_state is not None:
        random.seed(random_state)

    assert test_size > 0 and test_size < 1, \
        'test_size should be a float between (0, 1)'

    assert len(samples0) == len(samples1), \
        'number of samples in samples0, samples1 are not equal'

    idx_samples = list(range(len(samples0)))
    idx_test = random.sample(idx_samples, floor(test_size * len(samples0)))
    idx_train = list(set(idx_samples) - set(idx_test))

    # convert to np array for convenient array indexing
    samples0_np = np.stack([np.array(a) for a in samples0])
    samples1_np = np.stack([np.array(a) for a in samples1])

    samples_train0 = [(str(a[0]), int(a[1]))
                      for a in list(samples0_np[idx_train])]
    samples_test0 = [(str(a[0]), int(a[1]))
                     for a in list(samples0_np[idx_test])]
    samples_train1 = [(str(a[0]), int(a[1]))
                      for a in list(samples1_np[idx_train])]
    samples_test1 = [(str(a[0]), int(a[1]))
                     for a in list(samples1_np[idx_test])]

    assert len(samples_train0) == len(samples_train1), \
            'sample sizes not equal after split'
    assert len(samples_test0) == len(samples_test1), \
            'sample sizes not equal after split'

    return samples_train0, samples_test0, samples_train1, samples_test1


def train_test_split_views(views: list, 
                           test_size: float, 
                           random_state: bool = None) -> tuple[list, list]:
    if random_state is not None:
        random.seed(random_state)

    msg1 = 'test size should be a float between (0, 1)'
    assert test_size > 0 and test_size < 1, msg1

    msg2 = 'number of samples in views list is not equal'
    lens_views = set([len(view) for view in views])
    assert len(lens_views) == 1, msg2

    num_samples = len(views[0])
    idx_samples = list(range(num_samples))
    idx_test = random.sample(idx_samples, floor(test_size * num_samples))
    idx_train = list(set(idx_samples) - set(idx_test))

    samples_np = []
    for view in views:
        sample_np = np.stack([np.array(a) for a in view])
        samples_np.append(sample_np)
        
    train_views = []
    test_views = []
    for sample in samples_np:
        samples_train = [(str(a[0]), int(a[1]))
                         for a in list(sample[idx_train])]
        samples_test = [(str(a[0]), int(a[1]))
                        for a in list(sample[idx_test])]
        train_views.append(samples_train)
        test_views.append(samples_test)
    
    return train_views, test_views


def cascade_round(arr):
    s = 0.0
    arr_cp = np.zeros_like(arr)
    for i, a in enumerate(arr):
        s += a
        if s - (s // 1) > .5:
            arr_cp[i] = ceil(a)
        else:
            arr_cp[i] = floor(a)
    return arr_cp.astype(np.int32)


def cascade_round_subset(labels, percent):
    """
    labels: np.array of size (M, )
    
    return: mask of indicies to include if you want to respect stratification
    """
    unique, counts = np.unique(labels, return_counts=True)
    count_per_class = percent * counts
    # ok, but this is not exactly n% we will have some rounding to do here
    count_per_class = cascade_round(count_per_class)

    mask = np.hstack([
        np.random.choice(np.where(labels == unique[l])[0],
                         count_per_class[l],
                         replace=False) for l in range(unique.shape[0])
    ])

    return mask


def progressive_supset_sample(views: list, 
                              percent_unlbl: float,
                              percent_val: float,
                              k: float = 0.05,
                              random_state: int = 13) -> tuple[list, list, list]:
    """constructs training, validation, and unlabeled sets such that increasing
    the number of samples to use for training will maintain a subset / superset relation 
    on the training and validation sets; the training and validation sets are constructed
    by repeatedly sampling k% of the dataset and splitting the k%

    :param views: list of samples
    :type views: list
    :param percent_unlbl: percentage of samples to hold out as unlabeled data
    :type percent_unlbl: float
    :param percent_val: percentage of labeled samples to hold out for validation
    :type percent_val: float
    :param k: percentage to sample for one iteration, defaults to 0.05
    :type k: float, optional
    :param random_state: seed for random number generator, defaults to 13
    :type random_state: int, optional
    :return: training, validation, unlabeled lists
    :rtype: tuple[list, list, list]
    """
    msg1 = 'percent_unlbl should be a float between (0, 1)'
    assert percent_unlbl > 0 and percent_unlbl < 1, msg1

    msg2 = 'percent_val should be a float between (0, 1)'
    assert percent_val > 0 and percent_val < 1, msg2

    msg3 = 'number of samples in views list is not equal'
    lens_views = set(len(view) for view in views)
    assert len(lens_views) == 1, msg3

    num_samples = len(views[0])
    labels = np.array([l for _, l in views[0]])
    print(labels.shape)

    # number of iterations to sample k% of datapoints
    iters = round((1 - percent_unlbl) / k)
    print(f"iters: {iters}")

    mask_train = np.zeros(num_samples, dtype=bool)
    mask_val = np.zeros(num_samples, dtype=bool)
    for _ in range(iters):
        random.seed(random_state)
        np.random.seed(random_state)

        # retrieve indices for labeled data
        idx_labeled = cascade_round_subset(labels, k)

        random.seed(random_state) # paranoia

        # split labeled data into training / validation
        idx_val = random.sample(list(idx_labeled), floor(percent_val * len(idx_labeled)))
        idx_train = list(set(idx_labeled) - set(idx_val))

        mask_train[idx_train] = True
        mask_val[idx_val] = True

        mask_labels = np.ones(len(labels), dtype=bool)
        mask_labels[idx_labeled] = False
        labels = labels[mask_labels]

    mask_unlbl = np.ones(num_samples, dtype=bool)
    mask_unlbl[mask_train] = False
    mask_unlbl[mask_val] = False 
    
    train_views = []
    val_views = []
    unlbl_views = []
    for i in range(len(views)):
        view_np = np.stack([np.array(a) for a in views[i]])

        train_np = view_np[mask_train]
        val_np = view_np[mask_val]
        unlbl_np = view_np[mask_unlbl]

        train_view = [(str(a[0]), int(a[1]))
                         for a in list(train_np)]
        val_view = [(str(a[0]), int(a[1]))
                         for a in list(val_np)]
        unlbl_view = [(str(a[0]), int(a[1]))
                         for a in list(unlbl_np)]

        train_views.append(train_view)
        val_views.append(val_view)
        unlbl_views.append(unlbl_view)
    
    return train_views, val_views, unlbl_views

