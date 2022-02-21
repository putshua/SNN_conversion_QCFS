import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from torch import nn
import torch
from preprocess import GetImageNet
import os
import random
from Models.VGG import vgg16
from Models.ResNet import resnet34
from funcs import train_ann, seed_all
from utils import replace_activation_by_floor, replace_activation_by_neuron, replace_maxpool2d_by_avgpool2d
import numpy as np

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
    model = resnet34(num_classes=1000)
    model = replace_maxpool2d_by_avgpool2d(model)
    model = replace_activation_by_floor(model, t=args.l)

    model.cuda(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    train_ann(train, test, model, device=device, epochs=args.epochs, lr=args.lr, save=args.id, rank=rank)

    dist.destroy_process_group()

# multi processing
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpus', default=1, type=int, help='GPU number to use.')
    parser.add_argument('--batchsize', default=128, type=int, help='Batchsize')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--epochs', default=120, type=int, help='Training epochs')
    parser.add_argument('--id', default=None, type=str, help='Model identifier')
    parser.add_argument('--l', default=16, type=int, help='L')

    args = parser.parse_args()

    mp.spawn(main_worker, nprocs=args.gpus, args=(args.gpus, args))