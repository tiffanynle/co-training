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

def get_dataloaders(args):
    # load in training samples
    with open('cotraining_samples_lists_fixed.pkl', 'rb') as fp:
        dict = pickle.load(fp)
    data_labeled0 = dict['labeled']
    data_labeled1 = dict['inferred']
    
    # holdout -- 20% for a validation set
    p = 0.20
    idx_samples = list(range(len(data_labeled0)))
    idx_val = random.sample(idx_samples, floor(p * len(data_labeled0)))
    idx_train = list(set(idx_samples) - set(idx_val))

    lab0_samples = np.stack([np.array(a) for a in data_labeled0])
    lab1_samples = np.stack([np.array(a) for a in data_labeled1])

    samples_train0 = [(str(a[0]), int(a[1])) for a in list(lab0_samples[idx_train])]
    samples_val0 = [(str(a[0]), int(a[1])) for a in list(lab0_samples[idx_val])]
    samples_train1 = [(str(a[0]), int(a[1])) for a in list(lab1_samples[idx_train])]
    samples_val1 = [(str(a[0]), int(a[1])) for a in list(lab1_samples[idx_val])]

    # ResNet50 wants 224x224 images
    trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    # Make a dummy ImageFolder and update the class map
    data_train0 = datasets.ImageFolder(
                        '/ourdisk/hpc/ai2es/jroth/data/labeled', 
                        transform=trans)
    data_train0.class_to_idx = dict['class_map']
    data_train0.classes = list(dict['class_map'].keys())

    # More ImageFolders to get the train, validate loaders for both models
    data_train1 = copy.deepcopy(data_train0)
    data_val0 = copy.deepcopy(data_train0)
    data_val1 = copy.deepcopy(data_train0)
    
    data_train0.samples = samples_train0
    data_train1.samples = samples_train1
    data_val0.samples = samples_val0
    data_val1.samples = samples_val1

    # update root for train1, val1
    data_train1.root = '/ourdisk/hpc/ai2es'
    data_val1.root = '/ourdisk/hpc/ai2es'

    loader_train0 = DataLoader(data_train0, args.batch_size, shuffle=False)
    loader_val0 = DataLoader(data_val0, args.batch_size, shuffle=False)
    loader_train1 = DataLoader(data_train1, args.batch_size, shuffle=False)
    loader_val1 = DataLoader(data_val1, args.batch_size, shuffle=False)

    return loader_train0, loader_train1, loader_val0, loader_val1

def train(loader, model, loss_fn, optimizer, device):
    size = len(loader.dataset)
    num_batches = len(loader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        # Zero out the gradients (always...!)
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

def do_experiment(args, device):
    model0, model1 = resnet50().to(device), resnet50().to(device)
    loader_train0, loader_train1, loader_val0, loader_val1 = get_dataloaders(args)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer0 = SGD(model0.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer1 = SGD(model1.parameters(), lr=args.lr, momentum=args.momentum)
    
    scheduler0 = ReduceLROnPlateau(optimizer0)
    scheduler1 = ReduceLROnPlateau(optimizer1)

    print("training model0...")
    for t in range(args.epochs):
        print(f"epoch {t+1}\n------------------------------")
        train_loss = train(loader_train0, model0, loss_fn, optimizer0, device)
        val_loss = validate(loader_val0, model0, loss_fn, device)
        scheduler0.step(val_loss)
        
    print("\ntraining model1...")
    for t in range(args.epochs):
        print(f"epoch {t+1}\n------------------------------")
        train_loss = train(loader_train1, model1, loss_fn, optimizer1, device)
        val_loss = validate(loader_val1, model1, loss_fn, device)
        scheduler1.step(val_loss)  
        
def create_parser():
    parser = argparse.ArgumentParser(description='test with ResNet50')
    
    parser.add_argument('--epochs', type=int, default=10, 
                        help='training epochs (default: 10)')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate (default 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD (default 0.9')
    parser.add_argument('--gpu', action='store_true',
                        help='whether to use a GPU')
    
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    random.seed(13)
    device = torch.device("cuda" if args.gpu 
                          and torch.cuda.is_available()
                          else "cpu")
    print(f"using {device}")
    
    do_experiment(args, device)
    