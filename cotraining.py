import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.models import resnet50

import numpy as np

import argparse
import pickle
import copy
import random
from math import floor

from utils import add_to_imageloader, train_test_split_samples

# takes in a Tensor of shape e.g. (# instances, # prob outputs) and returns a tuple
# (Tensor[top probabilities], Tensor[predicted labels], Tensor[instance indexes])
def get_topk_pred(pred, k):
    prob, label = torch.max(pred, 1)
    idx = torch.argsort(prob, descending=True)[:k]
    return prob[idx].cpu(), label[idx].cpu(), idx.cpu()

def remove_collisions(lbl_model0, lbl_model1, idx_model0, idx_model1):
    # find instances and indices of instances that have
    # been labeled as most confident by both model0, model1
    inter, idx_inter0, idx_inter1 = np.intersect1d(
                                        idx_model0,
                                        idx_model1,
                                        return_indices=True)

    print(f"Number of predictions (model0): {len(idx_model0)}")
    print(f"Number of predictions (model1): {len(idx_model1)}")
    print(f"Found {len(inter)} potential conflicting predictions")

    # bool mask to identify the conflicting predictions (collision)
    mask_coll = lbl_model0[idx_inter0] != lbl_model1[idx_inter1]
    collisions = inter[mask_coll]

    print(f"Found {len(collisions)} conflicting predictions")

    if (len(collisions) > 0):
        print(f"Collisions: {collisions}")
        # find where these collisions are actually at
        # in their respective lists, and remove them...
        # (maybe want to return this as well? ...)
        idx_coll0 = idx_inter0[mask_coll]
        idx_coll1 = idx_inter1[mask_coll]

        # masks to remove the instances with conflicting predictions
        mask0 = np.ones(len(idx_model0), dtype=bool)
        mask0[idx_coll0] = False
        mask1 = np.ones(len(idx_model1), dtype=bool)
        mask1[idx_coll1] = False

        lbl_model0 = lbl_model0[mask0]
        lbl_model1 = lbl_model1[mask1]
        idx_model0 = idx_model0[mask0]
        idx_model1 = idx_model1[mask1]

    return lbl_model0, lbl_model1, idx_model0, idx_model1

# train two models on two different views
# then add top k% of predictions on the unlabeled set
# to the labeled datasets
def cotrain(loader0, loader1, loader_unlbl,
            model0, model1, loss_fn, optimizer0, optimizer1,
            k, device):

    # get top-k predictions (labels, instance indexes in the dataset)
    _, lbl_topk0, idx_topk0 = get_topk_pred(
                                    pred_model0,
                                    k if k <= len(pred_model0) else len(pred_model0))
    _, lbl_topk1, idx_topk1 = get_topk_pred(
                                    pred_model1, 
                                    k if k <= len(pred_model1) else len(pred_model1))

    print(f"Number of unlabeled instances: {len(loader_unlbl.dataset)}")

    # what if two models predict confidently on the same instance?
    # find and remove conflicting predictions from the lists
    # (may want to return the indices of the collisions too...?)
    lbl_topk0, lbl_topk1, idx_topk0, idx_topk1 = \
    remove_collisions(lbl_topk0, lbl_topk1, idx_topk0, idx_topk1)

    # convert from list to array for the convenient numpy indexing
    samples_unlbl = np.stack([np.array(a) for a in loader_unlbl.dataset.samples])
    list_samples0 = [(str(a[0]), int(a[1])) for a in list(samples_unlbl[idx_topk0])]
    list_samples1 = [(str(a[0]), int(a[1])) for a in list(samples_unlbl[idx_topk1])] 

    paths0 = [i for i, _ in list_samples0]
    paths1 = [i for i, _ in list_samples1]

    # add pseudolabeled instances to the labeled datasets
    loader0.dataset.samples = add_to_imagefolder(paths1, list(lbl_topk1), loader0.dataset)
    loader1.dataset.samples = add_to_imagefolder(paths0, list(lbl_topk0), loader1.dataset)

    # remove instances from unlabeled dataset
    mask_unlbl = np.ones(len(loader_unlbl.dataset), dtype=bool)
    mask_unlbl[idx_topk0] = False
    mask_unlbl[idx_topk1] = False
    print(f"Number of unlabeled instances to remove: {(~mask_unlbl).sum()}")
    samples_unlbl = samples_unlbl[mask_unlbl]
    list_unlbl = [(str(a[0]), int(a[1])) for a in list(samples_unlbl)]
    loader_unlbl.dataset.samples = list_unlbl

def train(loader, model, loss_fn, optimizer, device):
    size = len(loader.dataset)
    num_batches = len(loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        # Zero out the gradients
        optimizer.zero_grad()
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        
        if batch % 5 == 0:
            current = (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:5d} / {size:>5d}]")

    train_loss /= num_batches
    correct /= size

    print(f"train error:\n accuracy {(100*correct):>0.1f}%, avg loss: {train_loss:>8f}")

def validate(loader, model, loss_fn, device):
    size = len(loader.dataset)
    num_batches = len(loader)
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    val_loss /= num_batches
    correct /= size

    print(f"validation error:\n accuracy: {(100*correct):>0.1f}%, avg loss: {val_loss:>8f}")
    return val_loss

def main(args):
    with open('cotraining_samples_lists_fixed.pkl', 'rb') as fp:
        data = pickle.load(fp)

    # split samples into labeled, unlabeled (25/75 split)
    samples_train0, samples_train1, samples_unlbl0, samples_unlbl1 = \
    train_test_split_samples(data['labeled'], data['inferred'],
                             test_size=0.75, random_state=13)
    
    # split the data so we get 70/10/20 train/val/test
    samples_train0, samples_train1, samples_test0, samples_test1 = \
        train_test_split_samples(samples_train0, samples_train1,
                                 test_size=0.2, random_state=13)

    samples_train0, samples_train1, samples_val0, samples_val1 = \
        train_test_split_samples(samples_train0, samples_train1,
                                 test_size=0.125, random_state=13)

    trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
    ])

    # dataset and dataloader stuff here (I need 8 of them) ... what a mess
    data_train0 = datasets.ImageFolder('/ourdisk/hpc/ai2es/jroth/data/labeled', transform=trans)
    data_train0.class_to_idx = data['class_map']
    data_train0.classes = list(data['class_map'].keys())

    data_unlbl0 = copy.deepcopy(data_train0)
    data_unlbl0.samples = samples_unlbl0
    data_val0 = copy.deepcopy(data_train0)
    data_val0.samples = samples_val0
    data_test0 = copy.deepcopy(data_train0)
    data_test0.samples = samples_test0

    data_train1 = copy.deepcopy(data_train0)
    data_train1.root = '/ourdisk/hpc/ai2es'
    
    data_unlbl1 = copy.deepcopy(data_train1)
    data_unlbl1.samples = samples_unlbl1
    data_val1 = copy.deepcopy(data_train1)
    data_val1.samples = samples_val1
    data_test1 = copy.deepcopy(data_train1)
    data_test1.samples = samples_test1

    batch_size = 64
    loader_train0 = DataLoader(data_train0, batch_size, False)
    loader_unlbl0 = DataLoader(data_unlbl0, batch_size, False)
    loader_val0 = DataLoader(data_val0, batch_size, False)
    loader_test0 = DataLoader(data_test0, batch_size, False)
    
    loader_train1 = DataLoader(data_train1, batch_size, False)
    loader_unlbl1 = DataLoader(data_unlbl1, batch_size, False)
    loader_val1 = DataLoader(data_val1, batch_size, False)
    loader_test1 = DataLoader(data_test1, batch_size, False)

    device = torch.device("cuda" 
                          if torch.cuda.is_available()
                          else "cpu")
    print(f"using {device}")

    model0, model1 = resnet50().to(device), resnet50().to(device)
    
    loss_fn = nn.CrossEntropyLoss()
    
    optimizer0 = SGD(model0.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer1 = SGD(model1.parameters(), lr=args.lr, momentum=args.momentum)
    
    scheduler0 = ReduceLROnPlateau(optimizer0)
    scheduler1 = ReduceLROnPlateau(optimizer1)


    # co-training loop while there's still stuff in the unlabeled pools, probably
    while len(loader_unlbl0.dataset) > 0 and len(unloader_unlbl1.dataset) > 0:
        # train loop with validation, probably
        for t in range(args.epochs):
            pass
        for t in range(args.epochs):
            pass

    # print("training model0...")
    # for t in range(args.epochs):
    #     print(f"epoch {t+1}\n------------------------------")
    #     train(loader_train0, model0, loss_fn, optimizer0, device)
    #     val_loss = validate(loader_val0, model0, loss_fn, device)
    #     scheduler0.step(val_loss)
        
    # print("\ntraining model1...")
    # for t in range(args.epochs):
    #     print(f"epoch {t+1}\n------------------------------")
    #     train(loader_train1, model1, loss_fn, optimizer1, device)
    #     val_loss = validate(loader_val1, model1, loss_fn, device)
    #     scheduler1.step(val_loss)  
        
def create_parser():
    parser = argparse.ArgumentParser(description='co-training')
    
    parser.add_argument('-e', '--epochs', type=int, default=10, 
                        help='training epochs (default: 10)')
    parser.add_argument('-b', '--batch_size', type=int, default=64, 
                        help='batch size for training (default: 64)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        help='learning rate for SGD (default 1e-3)')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='momentum for SGD (default 0.9')
    #blah add more here
    
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    random.seed(13)
    
    main(args)
    