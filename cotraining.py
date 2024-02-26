import argparse
import functools
import gc
import os
import pickle
import random
from math import floor

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (enable_wrap,
                                         size_based_auto_wrap_policy, wrap)
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.models import resnet50

import metrics
import wandb
from ct_model import *
from DAHS.DAHB import DistributedAsynchronousGridSearch
from DAHS.torch_utils import sync_parameters
from utils import *


def create_model(auto_wrap_policy, device, num_classes, random_state=None):
    if random_state is not None:
        torch.manual_seed(random_state)

    model = resnet50(num_classes=num_classes).to(device)
    
    model = FSDP(model, 
                 auto_wrap_policy=auto_wrap_policy,
                )

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

    if rank == 0:
        print('Test error: \tCo-Accuracy: {:.2f}%'
              .format(100*test_acc))

    return test_acc


def training_process(args, rank, world_size):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    with open('cotraining_samples_lists_fixed.pkl', 'rb') as fp:
        data = pickle.load(fp)

    views = [data['labeled'], data['inferred']]

    samples_unlbl, samples_test = train_test_split_views(views, 
                                                             args.percent_test,
                                                             random_state=args.seed)
    samples_train, samples_val, samples_unlbl = progressive_supset_sample(samples_unlbl,
                                                                              args.percent_unlabeled,
                                                                              args.percent_val,
                                                                              args.k,
                                                                              random_state=args.seed)

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
    data_train0 = create_imagefolder(data, samples_train[0], dummy_path, trans)
    data_unlbl0 = create_imagefolder(data, samples_unlbl[0], dummy_path, trans)
    data_val0 = create_imagefolder(data, samples_val[0], dummy_path, trans)
    data_test0 = create_imagefolder(data, samples_test[0], dummy_path, trans)

    # Create ImageFolder objects for second view (we will also update the root/path)
    new_path = '/ourdisk/hpc/ai2es'
    data_train1 = create_imagefolder(data, samples_train[1], dummy_path, trans, new_path)
    data_unlbl1 = create_imagefolder(data, samples_unlbl[1], dummy_path, trans, new_path)
    data_val1 = create_imagefolder(data, samples_val[1], dummy_path, trans, new_path)
    data_test1 = create_imagefolder(data, samples_test[1], dummy_path, trans, new_path)

    train_views = [data_train0, data_train1]
    unlbl_views = [data_unlbl0, data_unlbl1]
    val_views =  [data_val0, data_val1]
    test_views = [data_test0, data_test1]

    num_classes = 3
    num_views = len(views)

    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=1_000_000)

    device = torch.device(rank % torch.cuda.device_count())
    torch.cuda.set_device(device)

    models = [create_model(auto_wrap_policy, device, num_classes) 
                for _ in range(num_views)]

    states = {'model0_state': models[0].state_dict(),
              'model1_state': models[1].state_dict()}

    ct_model = CoTrainingModel(rank, world_size, models)

    if rank == 0:
       wandb.init(project='co-training-calibrate',
                   entity='ai2es',
                   name=f'Co-Training',
                   config={'args': vars(args)})

    if rank == 0:
        print(vars(args))
        print("Length of datasets:\n Train: {} \tUnlabeled: {} \tVal: {} \tTest: {}"
            .format(len(samples_train[0]), len(samples_unlbl[0]),
                    len(samples_val[0]), len(samples_test[0])))

    loss_fn = nn.CrossEntropyLoss()

    # calibration metrics
    ece_criterion = metrics.ECELoss()
    ace_criterion = metrics.ACELoss()
    mace_criterion = metrics.MACELoss()

    k = floor(len(data_unlbl0) * args.k)
    if rank == 0:
        print(f"k: {k}")

    best_val_acc = 0.0
    best_val_loss = float('inf')
    c_iter_logs = []
    for c_iter in range(args.cotrain_iters):
        if len(data_unlbl0) == 0 and len(data_unlbl1) == 0:
            break
        if args.from_scratch and c_iter > 0:
            models = [create_model(auto_wrap_policy, device, num_classes) 
                        for _ in range(num_views)]
            ct_model = CoTrainingModel(rank, world_size, models)

        if rank == 0:
            print(f"co-training iteration: {c_iter}")
            print("train: {} unlabeled: {}"
                  .format(len(data_train0), len(data_unlbl0)))

        train_kwargs = {'device': device,
                        'iteration': c_iter,
                        'epochs': args.epochs,
                        'states': states,
                        'train_views': train_views,
                        'val_views': val_views,
                        'test_views': test_views,
                        'batch_size': args.batch_size}
        
        opt_kwargs = {'lr': args.learning_rate, 
                            'momentum': args.momentum}
        
        stopper_kwargs = {'metric': args.stopping_metric,
                           'patience': args.patience,
                           'min_delta': args.min_delta}
        
        best_val_acc_i, best_val_loss_i = ct_model.train(**train_kwargs, 
                                                         optimizer_kwargs=opt_kwargs, 
                                                         stopper_kwargs=stopper_kwargs)
        # update best val_acc
        best_val_acc = max(best_val_acc, best_val_acc_i)
        best_val_loss = min(best_val_loss, best_val_loss_i)

        # load best states for this iteration
        for i, model in enumerate(models):
            model.load_state_dict(states[f'model{i}_state'])

        # no persistent workers as we're only iterating through it once.
        sampler_val0, loader_val0 = create_sampler_loader(rank, world_size, data_val0, args.batch_size)
        sampler_test0, loader_test0 = create_sampler_loader(rank, world_size, data_test0, args.batch_size)
        sampler_val1, loader_val1 = create_sampler_loader(rank, world_size, data_val1, args.batch_size)
        sampler_test1, loader_test1 = create_sampler_loader(rank, world_size, data_test1, args.batch_size)
        
        # co-training val/test accuracy
        c_acc_val = c_test(rank, models[0], models[1], loader_val0, loader_val1, device)
        c_acc_test = c_test(rank, models[0], models[1], loader_test0, loader_test1, device)

        # prediction
        preds_softmax, labels = ct_model.predict(device, unlbl_views, num_classes, args.batch_size)

        if rank == 0:
            print("checking model calibration...")

        # calibration measurements
        calibration_logs = {}
        for i, preds in enumerate(preds_softmax):
            preds_np = preds.detach().cpu().numpy()
            lbls_np = labels.detach().cpu().numpy().astype(int)
            ece_loss = ece_criterion.loss(preds_np, lbls_np, logits=False)
            ace_loss = ace_criterion.loss(preds_np, lbls_np, logits=False)
            mace_loss = mace_criterion.loss(preds_np, lbls_np, logits=False)
            calibration_logs.update({f'ece_loss{i}': ece_loss,
                                     f'ace_loss{i}': ace_loss,
                                     f'mace_loss{i}': mace_loss})
        
        if rank == 0:
            print("calibrating models...")

        # recalibrate or something idk lol

        # update datasets
        ct_model.update(preds_softmax, train_views, unlbl_views, k)

        if rank == 0:
            print("testing after dataset update...")

        # test individual models after co-training update
        test_acc0, test_loss0 = test_ddp(rank, device, models[0], loader_test0, loss_fn)
        test_acc1, test_loss1 = test_ddp(rank, device, models[1], loader_test1, loss_fn)

        c_log = {'test_acc0': test_acc0,
                 'test_loss0': test_loss0,
                 'test_acc1': test_acc1,
                 'test_loss1': test_loss1,
                 'c_acc_val': c_acc_val,
                 'c_acc_test': c_acc_test}
        
        c_iter_logs += ct_model.logs
        c_iter_logs[-1] = ({**c_iter_logs[-1][0], **c_log, **calibration_logs}, c_iter_logs[-1][1])
        
    dist.barrier()

    # wandb.finish()

    return states, best_val_acc, c_iter_logs

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
    parser.add_argument('-md', '--min_delta', type=float, default=1e-3,
                        help='minimum delta for early stopping metric (default: %(default)s)')
    parser.add_argument('--cotrain_iters', type=int, default=100,
                        help='max number of iterations for co-training (default: %(default)s)')
    parser.add_argument('--k', type=float, default=[0.05],
                        help='percentage of unlabeled samples to bring in each \
                            co-training iteration (default: 0.025)')
    parser.add_argument('--percent_unlabeled', type=float, default=[0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05],
                        help='percentage of unlabeled samples to start with (default: 0.9')
    parser.add_argument('--percent_test', type=float, default=0.2,
                        help='percentage of samples to use for testing (default: %(default)s)')
    parser.add_argument('--percent_val', type=float, default=0.2,
                        help='percentage of labeled samples to use for validation (default: %(default)s)')
    parser.add_argument('--stopping_metric', type=str, default='accuracy', choices=['loss', 'accuracy'],
                        help='metric to use for early stopping (default: %(default)s)')
    parser.add_argument('--from_scratch', action='store_true',
                        help='whether to train a new model every co-training iteration (default: False)')
    parser.add_argument('--path', type=str, default='/ourdisk/hpc/ai2es/tiffanyle/co-training_hparams/calib_tests',
                        help='path for hparam search directory')
    parser.add_argument('--seed', type=int, default=13,
                        help='seed for random number generator (default: %(default)s)')
    parser.add_argument('--tag', type=str, help='tag for wandb run logging')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

    main(args, rank, world_size)
