import torch

import numpy as np

from math import floor
import random

# TODO write multi-view implementation (...)
def train_test_split_samples(samples0, samples1, test_size, random_state=None):
    if random_state is not None:
        random.seed(random_state)

    assert (test_size > 0 and test_size < 1,
        'test_size should be a float between (0, 1)')

    assert (len(samples0) == len(samples1),
        'number of samples in samples0, samples1 are not equal')

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

# TODO write multi-view implementation (will need some refactoring...)
def remove_collisions(lbl_model0, idx_model0, lbl_model1, idx_model1):
    # convert tensors to np arrays, as mixing use of tensors
    # and arrays together can cause some strange behaviors
    # (they will still both point to the same place in memory)
    lbl_model0_np = lbl_model0.cpu().detach().numpy()
    lbl_model1_np = lbl_model1.cpu().detach().numpy()
    idx_model0_np = idx_model0.cpu().detach().numpy()
    idx_model1_np = idx_model1.cpu().detach().numpy()
    
    # find which instances have been labeled with 
    # the most confidence by both model0 and model1
    inter, idx_inter0, idx_inter1 = np.intersect1d(
                                        idx_model0_np,
                                        idx_model1_np,
                                        return_indices=True)

    # which instances have a conflicting prediction from both models?
    mask_coll = lbl_model0_np[idx_inter0] != lbl_model1_np[idx_inter1]
    idx_colls = inter[mask_coll]

    if (len(idx_colls) > 0):
        print(f"Number of collisions: {len(idx_colls)}")
        idx_coll0 = idx_inter0[mask_coll]
        idx_coll1 = idx_inter1[mask_coll]
        
        mask = np.ones(len(idx_model0), dtype=bool)
        mask[idx_coll0] = False
        mask[idx_coll1] = False

        lbl_model0_np = lbl_model0_np[mask]
        idx_model0_np = idx_model0_np[mask]

        lbl_model1_np = lbl_model1_np[mask]
        idx_model1_np = idx_model1_np[mask]

    return lbl_model0_np, idx_model0_np, lbl_model1_np, idx_model1_np, idx_colls
