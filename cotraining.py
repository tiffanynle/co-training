import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import SGD
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
import gc
import argparse
import functools
import pickle
import random
from math import floor

from utils import *
from ct_utils import *
from ct_model import *
from DAHS.DAHB import DistributedAsynchronousGridSearch
from DAHS.torch_utils import sync_parameters

def create_model(auto_wrap_policy, device, num_classes):
    model = resnet50(num_classes=num_classes).to(device)
    
    model = FSDP(model, 
                 auto_wrap_policy=auto_wrap_policy,
                 mixed_precision=torch.distributed.fsdp.MixedPrecision(
                     param_dtype=torch.float16, 
                     reduce_dtype=torch.float32, 
                     buffer_dtype=torch.float32, 
                     cast_forward_inputs=True))

    return model


def c_test(rank, model0, model1, loader0, loader1, device):
    ddp_loss = torch.zeros(3).to(device)
    model0.eval()
    model1.eval()
    with torch.no_grad():
        for batch, ((X0, y0), (X1, y1))  in enumerate(zip(iter(loader0), iter(loader1))):
            X0, y0 = X0.to(device), y0.to(device)
            X1, y1 = X1.to(device), y1.to(device)

            output = torch.nn.Softmax(-1)(model0(X0)) * torch.nn.Softmax(-1)(model1(X1))

            ddp_loss[1] += (output.argmax(1) == y1).type(torch.float).sum().item()
            ddp_loss[2] += len(X0)
    
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)

    test_acc = ddp_loss[1] / ddp_loss[2] 
    test_loss = ddp_loss[0] / ddp_loss[2]

    if rank == 0:
        print('Test error: \tAccuracy: {:.2f}% \tAverage loss: {:.6f}'
              .format(100*test_acc, test_loss))

    return test_acc, test_loss

def cotraining_update(args, rank, world_size,
                      data_train0, data_train1,
                      data_unlbl0, data_unlbl1,
                      model0, model1, k, device):
    num_classes = len(data_train0.classes)

    sampler_unlbl0, loader_unlbl0 = create_sampler_loader(args, rank, world_size, data_unlbl0)
    sampler_unlbl1, loader_unlbl1 = create_sampler_loader(args, rank, world_size, data_unlbl1)

    pred_model0 = predict(world_size, args.batch_size, 
                          loader_unlbl0, model0, num_classes, device)[:len(data_unlbl0)]
    pred_model1 = predict(world_size, args.batch_size, 
                          loader_unlbl1, model1, num_classes, device)[:len(data_unlbl0)]
    
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

    # retrieve the instances that have been labeled with high confidence by the other model
    list_samples0 = [(str(a[0]), int(a[1])) for a in list(samples_unlbl0[idx_topk1])]
    list_samples1 = [(str(a[0]), int(a[1])) for a in list(samples_unlbl1[idx_topk0])]

    # image paths for both
    paths0 = [i for i, _ in list_samples0]
    paths1 = [i for i, _ in list_samples1]

    data_train0.samples = list(zip(paths0, lbl_topk1.tolist()))
    data_train1.samples = list(zip(paths1, lbl_topk0.tolist()))

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


def training_process(args, rank, world_size):
    random.seed(13)

    with open('cotraining_samples_lists_fixed.pkl', 'rb') as fp:
        data = pickle.load(fp)

    # split data first into not-test / test 
    samples_unlbl0, samples_test0, samples_unlbl1, samples_test1 = \
        train_test_split_samples(data['labeled'], data['inferred'],
                                 test_size=args.percent_test, random_state=13)

    # split data into labeled/unlabeled
    samples_train0, samples_unlbl0, samples_train1, samples_unlbl1 = \
        train_test_split_samples(samples_unlbl0, samples_unlbl1,
                             test_size=args.percent_unlabeled, random_state=13)

    # split train data into train/validation
    samples_train0, samples_val0, samples_train1, samples_val1 = \
        train_test_split_samples(samples_train0, samples_train1,
                                test_size=args.percent_val, random_state=13)

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

    num_classes = len(data_train0.classes)

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1_000_000)

    device = torch.device(rank % torch.cuda.device_count())
    torch.cuda.set_device(device)

    model0 = create_model(auto_wrap_policy, device, num_classes)
    model1 = create_model(auto_wrap_policy, device, num_classes)
    models = [model0, model1]

    sampler_val0, loader_val0 = create_sampler_loader(args, rank, world_size, data_val0)
    sampler_test0, loader_test0 = create_sampler_loader(args, rank, world_size, data_test0)

    sampler_val1, loader_val1 = create_sampler_loader(args, rank, world_size, data_val1)
    sampler_test1, loader_test1 = create_sampler_loader(args, rank, world_size, data_test1)

    if rank == 0:
        wandb.init(project='OO-CT-tests',
                    entity='ai2es',
                    name=f'Co-Training Test Run... 2!!!')

    loss_fn = nn.CrossEntropyLoss()

    k = floor(len(data_unlbl0) * args.k)
    best_val_acc = 0.0
    best_states = dict()
    c_iter_logs = []
    for c_iter in range(args.cotrain_iters):
        if len(data_unlbl0) == 0 and len(data_unlbl1) == 0:
            break
        if args.from_scratch and c_iter > 1:
            model0 = create_model(auto_wrap_policy, device, num_classes)
            model1 = create_model(auto_wrap_policy, num_classes)
            models = [model0, model1]
        
        print(f"co-training iteration: {c_iter}")

        train_views = [data_train0, data_train1]
        val_views = [data_val0, data_val1]
        ct_model = CoTrainingModel(models)
        states, best_val_acc_i = ct_model.train(rank, world_size, device,
                                                iteration=c_iter, 
                                                epochs=args.epochs,
                                                train_views = train_views, 
                                                val_views=val_views,
                                                batch_size=args.batch_size,
                                                optimizer_kwargs={'lr': args.learning_rate,
                                                                  'momentum': args.momentum})

        if (best_val_acc_i > best_val_acc):
            best_val_acc = best_val_acc_i
            best_states.update(states)

        for i, model in enumerate(models):
            model.load_state_dict(best_states[f'model{i}_state'])
        
        # co-training val/test accuracy
        c_acc_val, c_loss_val = c_test(rank, model0, model1, loader_val0, loader_val1, device)
        c_acc_test, c_loss_test = c_test(rank, model0, model1, loader_test0, loader_test1, device)

        # co-training update
        cotraining_update(args, rank, world_size, 
                          data_train0, data_train1, 
                          data_unlbl0, data_unlbl1, 
                          model0, model1, k, device)

        # test individual models after co-training update
        test_acc0, test_loss0 = test_ddp(rank, device, model0, loader_test0, loss_fn)
        test_acc1, test_loss1 = test_ddp(rank, device, model1, loader_test1, loss_fn)

        c_log = {'test_acc0': test_acc0,
                 'test_loss0': test_loss0,
                 'test_acc1': test_acc1,
                 'test_loss1': test_loss1,
                 'c_acc_val': c_acc_val,
                 'c_acc_test': c_acc_test}
        
        c_iter_logs += ct_model.logs
        c_iter_logs[-1] = ({**c_iter_logs[-1][0], **c_log}, c_iter_logs[-1][1])

        sampler_val0, loader_val0 = create_sampler_loader(args, rank, world_size, data_val0)
        sampler_test0, loader_test0 = create_sampler_loader(args, rank, world_size, data_test0)
        sampler_val1, loader_val1 = create_sampler_loader(args, rank, world_size, data_val1)
        sampler_test1, loader_test1 = create_sampler_loader(args, rank, world_size, data_test1)
        
    dist.barrier()

    # wandb.finish()

    return best_states, best_val_acc, c_iter_logs

def main(args, rank, world_size):

    setup(rank, world_size)
    
    search_space = ['k', 'percent_unlabeled']

    agent = sync_parameters(args, rank, search_space, DistributedAsynchronousGridSearch)

    args = agent.to_namespace(agent.combination)

    states, metric, logs = training_process(args, rank, world_size)

    if rank == 0:
        print('saving checkpoint')
        agent.save_checkpoint(states)
        for log, epoch in logs:
            wandb.log(log, step=epoch)
        wandb.finish()

    print('finishing combination')
    agent.finish_combination(float(metric))

    cleanup()


def create_parser():
    parser = argparse.ArgumentParser(description='co-training')
    
    parser.add_argument('-e', '--epochs', type=int, default=100, 
                        help='training epochs (default: %(default)s)')
    parser.add_argument('-b', '--batch_size', type=int, default=64, 
                        help='batch size for training (default: %(default)s)')
    parser.add_argument('-tb', '--test_batch_size', type=int, default=64, 
                        help=' batch size for testing (default: %(default)s)')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3,
                        help='learning rate for SGD (default: %(default)s)')
    parser.add_argument('-m', '--momentum', type=float, default=0.9,
                        help='momentum for SGD (default: %(default)s')
    parser.add_argument('-p', '--patience', type=float, default=32,
                        help='number of epochs to train for without improvement (default: %(default)s)')
    parser.add_argument('--cotrain_iters', type=int, default=100,
                        help='max number of iterations for co-training (default: %(default)s)')
    parser.add_argument('--k', type=float, default=[0.025],
                        help='percentage of unlabeled samples to bring in each \
                            co-training iteration (default: 0.025)')
    parser.add_argument('--percent_unlabeled', type=float, default=[0.95, 0.90, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.25, 0.2, 0.15, 0.1, 0.05],
                        help='percentage of unlabeled samples to start with (default: 0.9')
    parser.add_argument('--percent_test', type=float, default=0.2,
                        help='percentage of samples to use for testing (default: %(default)s)')
    parser.add_argument('--percent_val', type=float, default=0.2,
                        help='percentage of labeled samples to use for validation (default: %(default)s)')
    parser.add_argument('--stopping_metric', type=str, default='accuracy', choices=['loss', 'accuracy'],
                        help='metric to use for early stopping (default: %(default)s)')
    parser.add_argument('--from_scratch', action='store_true',
                        help='whether to train a new model every co-training iteration (default: False)')
    parser.add_argument('--path', type=str, default='/ourdisk/hpc/ai2es/tiffanyle/co-training_hparams/testing_oop',
                        help='path for hparam search directory')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    main(args, rank, world_size)