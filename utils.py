import numpy as np
import random
from math import floor
import torch

def add_to_imagefolder(paths, labels, dataset):
    """
    Adds the paths with the labels to an image classification dataset

    :list paths: a list of absolute image paths to add to the dataset
    :list labels: a list of labels for each path
    :Dataset dataset: the dataset to add the samples to
    """

    new_samples = list(zip(paths, labels))

    dataset.samples += new_samples

    return dataset.samples

# splits the datasets of the two views so that
# the instances inside are still aligned by index
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
    
    samples_train0 = [(str(a[0]), int(a[1])) for a in list(samples0_np[idx_train])]
    samples_test0 = [(str(a[0]), int(a[1])) for a in list(samples0_np[idx_test])]
    samples_train1 = [(str(a[0]), int(a[1])) for a in list(samples1_np[idx_train])]
    samples_test1 = [(str(a[0]), int(a[1])) for a in list(samples1_np[idx_test])]

    assert len(samples_train0) == len(samples_train1), 'sample sizes not equal after split'
    assert len(samples_test0) == len(samples_test1), 'sample sizes not equal after split'

    return samples_train0, samples_test0, samples_train1, samples_test1


class EarlyStopper:
    def __init__(self, stopping_metric, patience):
        self.stopping_metric = stopping_metric
        self.patience = patience
        self.epochs_since_improvement = 0
        self.stop = False 

        self.best_val_loss = float("inf")
        self.best_val_acc = 0

    # TODO perhaps there is a more elegant way to write this
    def is_new_best_metric(self, val_acc, val_loss):
        self.epochs_since_improvement += 1
        if self.stopping_metric == 'loss' and val_loss < self.best_val_loss - 1e-3:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.epochs_since_improvement = 0
            return True
        elif self.stopping_metric == 'accuracy' and val_acc > self.best_val_acc + 1e-3:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.epochs_since_improvement = 0
            return True
        return False
    
    def early_stop(self):
        if self.epochs_since_improvement > self.patience: 
            return True
        return False


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