import numpy as np
import random
from math import floor

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

    return samples_train0, samples_test0, samples_train1, samples_test1

