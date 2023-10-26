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
def get_topk_predictions(pred, k):
    prob, label = torch.max(pred, 1)
    idx = torch.argsort(prob, descending=True)[:k]
    return prob[idx], label[idx], idx


# TODO extract some of the logic into another function (so messy...)
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

    print('Intersection: {} \nidx_inter0: {} \nidx_inter1: {}'
          .format(inter, idx_inter0, idx_inter1))

    # which instances have a conflicting prediction from both models?
    mask_coll = lbl_model0_np[idx_inter0] != lbl_model1_np[idx_inter1]
    idx_colls = inter[mask_coll]

    # TODO may also want to return the predicted labels for the collisions
    if (len(idx_colls) > 0):
        print(f"idx_cols: {idx_colls}")
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


def predict(rank, world_size, batch_size, loader, model, num_classes, device):
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


# add top k% of predictions on the unlabeled datasets
# to the labeled datasets
def cotrain(args, rank, world_size,
            loader0, loader1, loader_unlbl0, loader_unlbl1,
            model0, model1, k, device):
    num_classes = len(loader0.dataset.classes)
    
    pred_model0 = predict(rank, world_size, args.batch_size,
                          loader_unlbl0, model0, num_classes, device)[:len(loader_unlbl0.dataset)]

    pred_model1 = predict(rank, world_size, args.batch_size,
                          loader_unlbl1, model1, num_classes, device)[:len(loader_unlbl0.dataset)]

    # this may be superfluous, but I want to make sure we agree.
    dist.broadcast(pred_model0, 0)
    dist.broadcast(pred_model1, 0)

    print("Number of unlabeled instances view0: {} view1: {}"
          .format(len(loader_unlbl0.dataset), len(loader_unlbl1.dataset)))
    
    _, lbl_topk0, idx_topk0 = get_topk_predictions(
                                    pred_model0,
                                    k if k <= len(pred_model0) 
                                    else len(pred_model0))
    _, lbl_topk1, idx_topk1 = get_topk_predictions(
                                    pred_model1, 
                                    k if k <= len(pred_model1) 
                                    else len(pred_model1))


    # if two models predict confidently on the same instance,
    # find and remove conflicting predictions from the lists
    lbl_topk0, idx_topk0, lbl_topk1, idx_topk1, idx_colls = \
    remove_collisions(lbl_topk0, idx_topk0, lbl_topk1, idx_topk1)

    samples_unlbl0 = np.stack([np.array(a) for a in loader_unlbl0.dataset.samples])
    samples_unlbl1 = np.stack([np.array(a) for a in loader_unlbl1.dataset.samples])

    if len(idx_colls) > 0:
        print("\nImage paths of collisions:\n unlbl0:\n {}\n unlbl1:\n {}\n"
             .format(samples_unlbl0[idx_colls],
                     samples_unlbl1[idx_colls]))

    # retrieve the instances that have been labeled with high confidence by the other model
    list_samples0 = [(str(a[0]), int(a[1])) for a in list(samples_unlbl0[idx_topk1])]
    list_samples1 = [(str(a[0]), int(a[1])) for a in list(samples_unlbl1[idx_topk0])]

    # image paths for both
    paths0 = [i for i, _ in list_samples0]
    paths1 = [i for i, _ in list_samples1]

    # add pseudolabeled instances to the labeled datasets
    data_labeled0 = add_to_imagefolder(paths0, lbl_topk1.tolist(), loader0.dataset)
    data_labeled1 = add_to_imagefolder(paths1, lbl_topk0.tolist(), loader1.dataset)
    loader0.dataset.samples = data_labeled0.samples
    loader1.dataset.samples = data_labeled1.samples

    # remove instances from unlabeled dataset
    mask = np.ones(len(loader_unlbl0.dataset), dtype=bool)
    mask[idx_topk0] = False
    mask[idx_topk1] = False

    print(f"Number of unlabeled instances to remove: {(~mask).sum()}")

    samples_unlbl0 = samples_unlbl0[mask]
    samples_unlbl1 = samples_unlbl1[mask]

    list_unlbl0 = [(str(a[0]), int(a[1])) for a in list(samples_unlbl0)]
    list_unlbl1 = [(str(a[0]), int(a[1])) for a in list(samples_unlbl1)]

    loader_unlbl0.dataset.samples = list_unlbl0
    loader_unlbl1.dataset.samples = list_unlbl1

    print(f"Remaining number of unlabeled instances: {len(loader_unlbl0.dataset)}")


def train(args, rank, world_size, loader, model, optimizer, epoch, device,
          sampler=None):
    if sampler:
        sampler.set_epoch(epoch)

    loss_fn = nn.CrossEntropyLoss()
    ddp_loss = torch.zeros(3).to(device)
    model.train()
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
    loss_fn = torch.nn.CrossEntropyLoss()
    ddp_loss = torch.zeros(3).to(device)
    model.eval()
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

def create_sampler_loader(args, rank, world_size, data, batch_size, cuda_kwargs, shuffle=True):
    sampler = DistributedSampler(data, rank=rank, num_replicas=world_size, shuffle=shuffle)

    loader_kwargs = {'batch_size': batch_size, 'sampler': sampler}
    loader_kwargs.update(cuda_kwargs)

    loader = DataLoader(data, **loader_kwargs)

    return sampler, loader

class EarlyStopper:
    def __init__(self, stopping_metric, patience):
        self.stopping_metric = stopping_metric
        self.patience = patience
        self.epochs_since_improvement = 0

        self.best_val_loss = float("inf")
        self.best_val_acc = 0

    # TODO surely there is a nicer way to write this
    def update_new_best_metric(self, val_acc, val_loss):
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

# TODO clean this
def train_wrapper(args, rank, world_size, 
                  loader_train, loader_val, loader_test, sampler_train, 
                  model, optimizer, scheduler, stopper, 
                  states, states_keys, log_keys, device):
    for epoch in range(1, args.epochs + 1):
        train(args, rank, world_size, loader_train, model, optimizer, epoch, device, sampler_train)
        val_acc, val_loss = test(args, rank, world_size, loader_val, model, device)
        test_acc, test_loss = test(args, rank, world_size, loader_test, model, device)

        if stopper.update_new_best_metric(val_acc, val_loss):
            states_values = [model.state_dict(), optimizer.state_dict()]
            update_dict_kv_pairs(states, states_keys, states_values)

        if rank == 0:
            log_vals = [val_acc, val_loss, 
                        stopper.best_val_acc, stopper.best_val_loss, 
                        test_acc, test_loss]
            log = create_wandb_log(log_keys, log_vals)
            wandb.log(log, step=epoch if 'model0_state' in states_keys else epoch + args.epochs)

        if stopper.early_stop():
            break

        scheduler.step(val_loss)


def update_dict_kv_pairs(d, dkeys, dvals):
    assert len(dkeys) == len(dvals), \
    "Number of keys, values to update in dict are not equal"
    for k, v in zip(dkeys, dvals):
        d[k] = v

def create_wandb_log(log_keys, log_vals):
    assert len(log_keys) == len(log_vals), \
    "Number of keys, values are not equal"
    d = dict()
    for k, v in zip(log_keys, log_vals):
        d[k] = v
    return d

def create_models(args, auto_wrap_policy, device):
       # instantiate models, send to device 
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

    return model0, model1, optimizer0, optimizer1, scheduler0, scheduler1

def training_process(args, rank, world_size):
    random.seed(13)

    with open('cotraining_samples_lists_fixed.pkl', 'rb') as fp:
        data = pickle.load(fp)

    # split data into labeled/unlabeled
    samples_train0, samples_unlbl0, samples_train1, samples_unlbl1 = \
        train_test_split_samples(data['labeled'], data['inferred'],
                                test_size=args.percent_unlabeled, random_state=13)
        
    # split the data so we get 70/10/20 train/val/test
    samples_train0, samples_test0, samples_train1, samples_test1 = \
        train_test_split_samples(samples_train0, samples_train1,
                                 test_size=0.2, random_state=13)

    samples_train0, samples_val0, samples_train1, samples_val1 = \
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
        size_based_auto_wrap_policy, min_num_params=1_000_000
    )

    device = torch.device(rank % torch.cuda.device_count())
    torch.cuda.set_device(device)

    model0, model1, optimizer0, optimizer1, scheduler0, scheduler1 = create_models(args, auto_wrap_policy, device)

    cuda_kwargs = {'num_workers': 12, 'pin_memory': True, 'shuffle': False}

    states = {
        'model0_state': model0.state_dict(), 
        'optimizer0_state': optimizer0.state_dict(),
        'model1_state': model1.state_dict(),
        'optimizer1_state': optimizer1.state_dict()}

    # ...    
    states_keys0 = ['model0_state', 'optimizer0_state']
    states_keys1 = ['model1_state', 'optimizer1_state']

    # ...
    log_keys0 = ['val_acc0', 'val_loss0', 'best_val_acc0', 'best_val_loss0', 'test_acc0', 'test_loss0']
    log_keys1 = ['val_acc1', 'val_loss1', 'best_val_acc1', 'best_val_loss1', 'test_acc1', 'test_loss1']

    # Instantiate samplers and get DataLoader objects
    sampler_train0, loader_train0 = create_sampler_loader(args, rank, world_size, data_train0, args.batch_size, cuda_kwargs)
    sampler_unlbl0, loader_unlbl0 = create_sampler_loader(args, rank, world_size, data_unlbl0, args.batch_size, cuda_kwargs, shuffle=False)
    sampler_val0, loader_val0 = create_sampler_loader(args, rank, world_size, data_val0, args.batch_size, cuda_kwargs)
    sampler_test0, loader_test0 = create_sampler_loader(args, rank, world_size, data_test0, args.test_batch_size, cuda_kwargs)

    sampler_train1, loader_train1 = create_sampler_loader(args, rank, world_size, data_train1, args.batch_size, cuda_kwargs)
    sampler_unlbl1, loader_unlbl1 = create_sampler_loader(args, rank, world_size, data_unlbl1, args.batch_size, cuda_kwargs, shuffle=False)
    sampler_val1, loader_val1 = create_sampler_loader(args, rank, world_size, data_val1, args.batch_size, cuda_kwargs)
    sampler_test1, loader_test1 = create_sampler_loader(args, rank, world_size, data_test1, args.test_batch_size, cuda_kwargs)

    k = floor(len(data_unlbl0) * args.k)
    for c_iter in range(1, args.cotrain_iters + 1):        
        # if there's no more unlabeled examples for co-training, stop the process
        if len(loader_unlbl0.dataset) == 0 and len(loader_unlbl1.dataset) == 0: 
            break

        if rank == 0:
            wandb.init(project='Co-Training', entity='ai2es',
            name=f"{rank}: Co-Training - {c_iter}",
            config={'args': vars(args)})
            print(f"Co-Training Iteration: {c_iter}\n---------------------------------------")

#        if args.from_scratch and c_iter > 1:
#            model0, model1, optimizer0, optimizer1, scheduler0, scheduler1 = create_models(args, auto_wrap_policy, device)

        stopper0 = EarlyStopper(stopping_metric=args.stopping_metric, patience=args.patience)
        stopper1 = EarlyStopper(stopping_metric=args.stopping_metric, patience=args.patience)

        train_wrapper(args, rank, world_size, 
                      loader_train0, loader_val0, loader_test0, sampler_train0,
                      model0, optimizer0, scheduler0, stopper0,
                      states, states_keys0, log_keys0, device)

        train_wrapper(args, rank, world_size, 
                      loader_train1, loader_val1, loader_test1, sampler_train1,
                      model1, optimizer1, scheduler1, stopper1,
                      states, states_keys1, log_keys1, device)

        # load the best states
        model0.load_state_dict(states['model0_state'])
        model1.load_state_dict(states['model1_state'])
        
        cotrain(args, rank, world_size,
                loader_train0, loader_train1, 
                loader_unlbl0, loader_unlbl1, 
                model0, model1, k,device)

        if rank == 0:
            wandb.finish()

        # Re-instantiate samplers and get DataLoader objects
        sampler_train0, loader_train0 = create_sampler_loader(args, rank, world_size, data_train0, args.batch_size, cuda_kwargs)
        sampler_unlbl0, loader_unlbl0 = create_sampler_loader(args, rank, world_size, data_unlbl0, args.batch_size, cuda_kwargs, shuffle=False)
        sampler_val0, loader_val0 = create_sampler_loader(args, rank, world_size, data_val0, args.batch_size, cuda_kwargs)
        sampler_test0, loader_test0 = create_sampler_loader(args, rank, world_size, data_test0, args.test_batch_size, cuda_kwargs)

        sampler_train1, loader_train1 = create_sampler_loader(args, rank, world_size, data_train1, args.batch_size, cuda_kwargs)
        sampler_unlbl1, loader_unlbl1 = create_sampler_loader(args, rank, world_size, data_unlbl1, args.batch_size, cuda_kwargs, shuffle=False)
        sampler_val1, loader_val1 = create_sampler_loader(args, rank, world_size, data_val1, args.batch_size, cuda_kwargs)
        sampler_test1, loader_test1 = create_sampler_loader(args, rank, world_size, data_test1, args.test_batch_size, cuda_kwargs)

    dist.barrier()

    # TODO return states, best metric
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
    
    parser.add_argument('-e', '--epochs', type=int, default=100, 
                        help='training epochs (default: 100)')
    parser.add_argument('-b', '--batch_size', type=int, default=64, 
                        help='batch size for training (default: 64)')
    parser.add_argument('-tb', '--test_batch_size', type=int, default=64, 
                        help=' batch size for testing (default: 64)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        help='learning rate for SGD (default: 1e-3)')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='momentum for SGD (default: 0.9')
    parser.add_argument('-p', '--patience', type=float, default=32,
                        help='number of epochs to train for without improvement (default: 32)')
    parser.add_argument('--cotrain_iters', type=int, default=100,
                        help='max number of iterations for co-training (default: 100)')
    parser.add_argument('--k', type=float, default=0.025,
                        help='percentage of unlabeled samples to bring in each \
                            co-training iteration (default: 0.025)')
    parser.add_argument('--percent_unlabeled', type=float, default=0.75,
                        help='percentage of unlabeled samples to start with (default: 0.75)')
    parser.add_argument('--stopping_metric', type=str, default='accuracy', choices=['loss', 'accuracy'],
                        help='metric to use for early stopping (default: %(default)s)')
    parser.add_argument('--from_scratch', action='store_true',
                        help='whether to train a new model every co-training iteration (default: False)')
    # TODO add additional arguments
    
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    main(args, rank, world_size)
    
