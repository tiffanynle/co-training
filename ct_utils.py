import torch

import numpy as np

from math import floor
import random

from utils import *

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

    assert (len(samples_train0) == len(samples_train1),
            'sample sizes not equal after split')
    assert (len(samples_test0) == len(samples_test1),
            'sample sizes not equal after split')

    return samples_train0, samples_test0, samples_train1, samples_test1