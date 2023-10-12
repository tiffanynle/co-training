import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from torchvision import datasets, transforms
from torchvision.models import resnet50

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

import numpy as np
import wandb

import os
import argparse
import functools
import pickle
import copy
import random
from math import floor

from utils import add_to_imagefolder, train_test_split_samples

# takes in a Tensor of shape e.g. (# instances, # prob outputs) and returns a tuple
# (Tensor[top probabilities], Tensor[predicted labels], Tensor[instance indexes])
def get_topk_pred(pred, k):
    prob, label = torch.max(pred, 1)
    idx = torch.argsort(prob, descending=True)[:k]
    return prob[idx].cpu(), label[idx].cpu(), idx.cpu()

# TODO need to revise this
# once data starts getting removed from the unlabeled sets
# it is likely that the two views aren't aligned anymore (...)
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
        idx_model0 = idx_model0[mask0]

        lbl_model1 = lbl_model1[mask1]
        idx_model1 = idx_model1[mask1]

    return lbl_model0, lbl_model1, idx_model0, idx_model1

def predict(loader, model, num_classes, rank, world_size, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for X, y in loader:
            tensor_list = [torch.zeros((X.shape[0], num_classes), dtype=torch.float16)
                           .to(device) 
                           for _ in range(world_size)]
            dist.all_gather(tensor_list, model(X.to(device)))
            batch_outputs = torch.cat(tensor_list)
            predictions.append(batch_outputs)
    
    return torch.cat(predictions) # output shape (# instances, # outputs)

# add top k% of predictions on the unlabeled datasets
# to the labeled datasets
def cotrain(loader0, loader1, loader_unlbl0, loader_unlbl1,
            model0, model1, k,
            rank, world_size, device):
    num_classes = len(loader0.dataset.classes)
    pred_model0 = predict(loader_unlbl0, model0, num_classes, rank, world_size, device)
    pred_model1 = predict(loader_unlbl1, model1, num_classes, rank, world_size, device)

    # get top-k predictions (labels, instance indexes in the dataset)
    _, lbl_topk0, idx_topk0 = get_topk_pred(
                                    pred_model0,
                                    k if k <= len(pred_model0) 
                                    else len(pred_model0))

    _, lbl_topk1, idx_topk1 = get_topk_pred(
                                    pred_model1, 
                                    k if k <= len(pred_model1) 
                                    else len(pred_model1))

    print(f"Number of unlabeled instances: {len(loader_unlbl0.dataset)}")

    # what if two models predict confidently on the same instance?
    # find and remove conflicting predictions from the lists
    # TODO also return the indices of the conflicting predictions
    # and print out the imagepaths, maybe?
    lbl_topk0, lbl_topk1, idx_topk0, idx_topk1 = \
    remove_collisions(lbl_topk0, lbl_topk1, idx_topk0, idx_topk1)

    # convert from list to array for the convenient numpy indexing
    samples_unlbl0 = np.stack([np.array(a) for a in loader_unlbl0.dataset.samples])
    samples_unlbl1 = np.stack([np.array(a) for a in loader_unlbl1.dataset.samples])

    list_samples0 = [(str(a[0]), int(a[1])) for a in list(samples_unlbl0[idx_topk0])]
    list_samples1 = [(str(a[0]), int(a[1])) for a in list(samples_unlbl1[idx_topk1])] 

    paths0 = [i for i, _ in list_samples0]
    paths1 = [i for i, _ in list_samples1]

    # add pseudolabeled instances to the labeled datasets
    loader0.dataset.samples = add_to_imagefolder(paths1, lbl_topk1.tolist(), loader0.dataset)
    loader1.dataset.samples = add_to_imagefolder(paths0, lbl_topk0.tolist(), loader1.dataset)

    # remove instances from unlabeled dataset
    mask_unlbl0 = np.ones(len(loader_unlbl0.dataset), dtype=bool)
    mask_unlbl1 = np.ones(len(loader_unlbl1.dataset), dtype=bool)

    mask_unlbl0[idx_topk0] = False
    mask_unlbl1[idx_topk1] = False

    print(f"Number of unlabeled instances to remove: {(~mask_unlbl0).sum()}")

    samples_unlbl0 = samples_unlbl0[mask_unlbl0]
    samples_unlbl1 = samples_unlbl1[mask_unlbl1]

    list_unlbl0 = [(str(a[0]), int(a[1])) for a in list(samples_unlbl0)]
    list_unlbl1 = [(str(a[0]), int(a[1])) for a in list(samples_unlbl1)]

    loader_unlbl0.dataset.samples = list_unlbl0
    loader_unlbl1.dataset.samples = list_unlbl1

    print("Size of unlbl0: {}\t unlbl1: {}\t lbl0: {}\t lbl1: {}\t"
        .format(len(loader_unlbl0.dataset), len(loader_unlbl1.dataset),
                len(loader0.dataset), len(loader1.dataset)))

def train(args, rank, world_size, loader, model, optimizer, epoch, device,
          sampler=None):
    if sampler:
        sampler.set_epoch(epoch)

    model.train()
    loss_fn = nn.CrossEntropyLoss()
    ddp_loss = torch.zeros(3).to(device)
    for batch, (X, y) in enumerate(loader):
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

    if rank == 0:
        print('Train Epoch: {} \tAccuracy: {:.2f}% \tAverage Loss: {:.6f}'
              .format(epoch, 
                      100*(ddp_loss[1] / ddp_loss[2]), 
                      ddp_loss[0] / ddp_loss[2]))

def test(args, rank, world_size, loader, model, device):
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss()
    ddp_loss = torch.zeros(3).to(device)
    with torch.no_grad():
        for batch, (X, y) in enumerate(loader):
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

def create_imagefolder(data, samples, path, transform, new_path=None):
    imgfolder = datasets.ImageFolder(path, transform=transform)
    imgfolder.class_to_idx = data['class_map']
    imgfolder.classes = list(data['class_map'].keys())
    imgfolder.samples = samples

    if new_path is not None:
        imgfolder.root = new_path

    return imgfolder

def create_sampler_loader(args, rank, world_size, data, cuda_kwargs):
    sampler = DistributedSampler(data, rank=rank, num_replicas=world_size, shuffle=True)

    loader_kwargs = {'batch_size': args.batch_size, 'sampler': sampler}
    loader_kwargs.update(cuda_kwargs)

    loader = DataLoader(data, **loader_kwargs)

    return sampler, loader    

def training_process(args, rank, world_size):
    random.seed(13)

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
    if rank == 0:
        print("Length of datasets:\n Train: {} \tUnlabeled: {} \tVal: {} \tTest: {}"
            .format(len(samples_train0), len(samples_unlbl0),
                    len(samples_val0), len(samples_test0)))

    # ResNet50 wants 224x224 images
    trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
    ])

    # Create ImageFolder objects for first view
    dummy_path = '/ourdisk/hpc/ai2es/jroth/data/labeled' 
    data_train0 = create_imagefolder(data, samples_train0, dummy_path, trans)
    data_unlbl0 = create_imagefolder(data, samples_unlbl0, dummy_path, trans)
    data_val0 = create_imagefolder(data, samples_val0, dummy_path, trans)
    data_test0 = create_imagefolder(data, samples_test0, dummy_path, trans)

    # Create ImageFolder objects for second view (we will also update the root/path)
    new_path = '/ourdisk/hpc/ai2es'
    data_train1 = create_imagefolder(data, samples_train1, dummy_path, trans, new_path)
    data_unlbl1 = create_imagefolder(data, samples_unlbl1, dummy_path, trans, new_path)
    data_val1 = create_imagefolder(data, samples_val1, dummy_path, trans, new_path)
    data_test1 = create_imagefolder(data, samples_test1, dummy_path, trans, new_path)

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=10_000_000
    )

    device = torch.device(rank % torch.cuda.device_count())
    torch.cuda.set_device(device)

    # instantiate models, send to device 
    # (make sure to set the number of classes to predict... default is 1000)
    model0, model1 = resnet50(num_classes=3).to(device), resnet50(num_classes=3).to(device)
    
    # wrapping to take advantage of FSDP
    model0 = FSDP(model0, 
                 auto_wrap_policy=auto_wrap_policy,
                 mixed_precision=torch.distributed.fsdp.MixedPrecision(
                     param_dtype=torch.float16, 
                     reduce_dtype=torch.float32, 
                     buffer_dtype=torch.float32, 
                     cast_forward_inputs=True)
                    )

    model1 = FSDP(model1, 
                 auto_wrap_policy=auto_wrap_policy,
                 mixed_precision=torch.distributed.fsdp.MixedPrecision(
                     param_dtype=torch.float16, 
                     reduce_dtype=torch.float32, 
                     buffer_dtype=torch.float32, 
                     cast_forward_inputs=True)
                     )

    optimizer0 = optim.SGD(model0.parameters(), lr=args.learning_rate, momentum=args.momentum)
    optimizer1 = optim.SGD(model1.parameters(), lr=args.learning_rate, momentum=args.momentum)
    
    scheduler0 = ReduceLROnPlateau(optimizer0)
    scheduler1 = ReduceLROnPlateau(optimizer1)

    cuda_kwargs = {'num_workers': 12, 'pin_memory': True, 'shuffle': False}

    states = {
        'model0_state': model0.state_dict(), 
        'optimizer0_state': optimizer0.state_dict(),
        'model1_state': model1.state_dict(),
        'optimizer1_state': optimizer1.state_dict()}


    k = floor(len(data_unlbl0) * args.k)
    best_vloss0, best_vloss1 = float("inf"), float("inf")
    epochs_since_improvement0 = 0
    epochs_since_improvement1 = 0
    for c_iter in range(1, args.cotrain_iters + 1):
        if rank == 0:
            wandb.init(project='Co-Training', entity='ai2es',
            name=f"{rank}: Co-Training",
            config={'args': vars(args)})
            print(f"Co-Training Iteration: {c_iter}\n---------------------------------------")

        # Instantiate samplers and get DataLoader objects
        sampler_train0, loader_train0 = create_sampler_loader(args, rank, world_size, data_train0, cuda_kwargs)
        sampler_unlbl0, loader_unlbl0 = create_sampler_loader(args, rank, world_size, data_unlbl0, cuda_kwargs)
        sampler_val0, loader_val0 = create_sampler_loader(args, rank, world_size, data_val0, cuda_kwargs)
        sampler_test0, loader_test0 = create_sampler_loader(args, rank, world_size, data_test0, cuda_kwargs)

        sampler_train1, loader_train1 = create_sampler_loader(args, rank, world_size, data_train1, cuda_kwargs)
        sampler_unlbl1, loader_unlbl1 = create_sampler_loader(args, rank, world_size, data_unlbl1, cuda_kwargs)
        sampler_val1, loader_val1 = create_sampler_loader(args, rank, world_size, data_val1, cuda_kwargs)
        sampler_test1, loader_test1 = create_sampler_loader(args, rank, world_size, data_test1, cuda_kwargs)

        # if there's no more unlabeled examples for co-training, stop the process
        if len(loader_unlbl0.dataset) == 0 and len(loader_unlbl1.dataset) == 0: 
            break
        
        for epoch in range(1, args.epochs + 1):
            train(args, rank, world_size, loader_train0, model0, optimizer0, epoch, device, sampler_train0)
            val_acc0, vloss0 = test(args, rank, world_size, loader_val0, model0, device)

            epochs_since_improvement0 += 1
            if vloss0 < best_vloss0 - 1e-3: # TODO should this be accuracy or loss...?
                best_vloss0 = vloss0
                epochs_since_improvement0 = 0
                states['model0_state'] = model0.state_dict()
                states['optimizer0_state'] = optimizer0.state_dict()
            
            if rank == 0:
                wandb.log({'val_acc0': val_acc0,
                          'vloss0': vloss0,
                          'best_vloss0': best_vloss0},
                          step=epoch)
                
            if epochs_since_improvement0 > args.patience: 
                break

            scheduler0.step(vloss0)
        
        for epoch in range(1, args.epochs + 1):
            train(args, rank, world_size, loader_train1, model1, optimizer1, epoch, device, sampler_train1)
            val_acc1, vloss1 = test(args, rank, world_size, loader_val1, model1, device)

            epochs_since_improvement1 += 1
            if vloss1 < best_vloss1 - 1e-3:
                best_vloss1 = vloss1
                epochs_since_improvement1 = 0
                states['model1_state'] = model1.state_dict()
                states['optimizer1_state'] = optimizer1.state_dict()
            
            if rank == 0:
                wandb.log({'val_acc1': val_acc1,
                          'vloss1': vloss1,
                          'best_vloss1': best_vloss1},
                          step=epoch + args.epochs + 1)
                
            if epochs_since_improvement1 > args.patience: 
                break

            scheduler1.step(vloss1)
        
        cotrain(loader_train0, loader_train1, 
                loader_unlbl0, loader_unlbl1, 
                model0, model1, k,
                rank, world_size, device)

        # test here, probably
        test_acc0, test_loss0 = test(args, rank, world_size, loader_test0, model0, device)
        test_acc1, test_loss1 = test(args, rank, world_size, loader_test1, model1, device)

        if rank == 0:
            wandb.log({'test_acc0': test_acc0,
                       'test_loss0': test_loss0,
                       'test_acc1' : test_acc1,
                       'test_loss1': test_loss1},
                       step=None)

        if rank == 0:
            wandb.finish()

    # barrier or something
    dist.barrier()

    # TODO return dict of states, metrics <-- (what is this...?)
    return states, None

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(args, rank, world_size):

    setup(rank, world_size)

    states, metrics = training_process(args, rank, world_size)

    if rank == 0:
        torch.save(states, '/home/scratch/tiffanyle/cotraining/states.pth')

    cleanup()
        
def create_parser():
    parser = argparse.ArgumentParser(description='co-training')
    
    parser.add_argument('-e', '--epochs', type=int, default=10, 
                        help='training epochs (default: 10)')
    parser.add_argument('-b', '--batch_size', type=int, default=64, 
                        help='batch size for training (default: 64)')
    parser.add_argument('-tb', '--test_batch_size', type=int, default=64, 
                        help='test batch size for training (default: 64)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        help='learning rate for SGD (default 1e-3)')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='momentum for SGD (default 0.9')
    parser.add_argument('-p', '--patience', type=float, default=10,
                        help='number of epochs to train for without improvement (default: 10)')
    parser.add_argument('--cotrain_iters', type=int, default=25,
                        help='max number of iterations for co-training (default: 25)')
    parser.add_argument('--k', type=float, default=0.05,
                        help='percentage of unlabeled samples to bring in each \
                            co-training iteration (default: 0.05)')
    #blah add more here
    
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    main(args, rank, world_size)
    