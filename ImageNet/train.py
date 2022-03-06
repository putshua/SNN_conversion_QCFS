import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from torch import nn
import torch
from Models import modelpool
from Preprocess.getdataloader import GetImageNet
from funcs import train_ann, seed_all
from utils import replace_activation_by_floor, replace_activation_by_neuron, replace_maxpool2d_by_avgpool2d
import os

def main_worker(rank, gpus, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='nccl', rank=rank, world_size=gpus)
    
    device=f'cuda:{rank}'
    torch.cuda.set_device(device)
    seed_all()

    batchsize = int(args.batchsize / gpus)
    train, test = GetImageNet(batchsize)

    # model preparing
    model = modelpool(args.model)
    model = replace_maxpool2d_by_avgpool2d(model)
    model = replace_activation_by_floor(model, t=args.l)

    criterion = nn.CrossEntropyLoss()

    model.cuda(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    train_ann(train, test, model, args.epochs, device, criterion, args.lr, args.wd, args.id, rank, True)
    
    dist.destroy_process_group()