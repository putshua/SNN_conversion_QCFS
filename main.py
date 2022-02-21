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
from funcs import train_ann, eval_snn, eval_ann
from utils import replace_activation_by_floor, replace_activation_by_neuron, replace_maxpool2d_by_avgpool2d
import numpy as np

def main_worker(rank, gpus, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='nccl', rank=rank, world_size=gpus)
    
    device=f'cuda:{rank}'
    torch.cuda.set_device(device)
    random.seed(42)
    torch.manual_seed(42)

    batchsize = int(args.batchsize / gpus)
    train, test = GetImageNet(batchsize)

    # model preparing
    model = resnet34(num_classes=1000)
    model = replace_maxpool2d_by_avgpool2d(model)
    model = replace_activation_by_floor(model, t=args.t)

    model.cuda(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    train_ann(train, test, model, device=device, epochs=args.epochs, lr=args.lr, save=args.id, rank=rank)

    dist.destroy_process_group()

def main_tester(rank, gpus, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='gloo', rank=rank, world_size=gpus)
    
    device=f'cuda:{rank}'
    torch.cuda.set_device(device)
    random.seed(42)
    torch.manual_seed(42)

    batchsize = int(args.batchsize / gpus)
    train, test = GetImageNet(batchsize)

    # model preparing
    model = resnet34(num_classes=1000)
    model = replace_maxpool2d_by_avgpool2d(model)
    model = replace_activation_by_floor(model, t=16)
    
    model.load_state_dict(torch.load('./saved/'+args.id+'.pth'))

    model = replace_activation_by_neuron(model)
    
    model.cuda(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    acc = eval_snn(test, model, device=device, sim_len=args.t, rank=rank)

    dist.all_reduce(acc)
    acc/=50000
    if rank == 0:
        print(acc[15], acc[31], acc[63], acc[127], acc[255], acc[511])

    dist.destroy_process_group()

def main_anntester(rank, gpus, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='gloo', rank=rank, world_size=gpus)
    
    device=f'cuda:{rank}'
    torch.cuda.set_device(device)
    random.seed(42)
    torch.manual_seed(42)

    batchsize = int(args.batchsize / gpus)
    train, test = GetImageNet(batchsize)

    # model preparing
    model = resnet34(num_classes=1000)
    model = replace_maxpool2d_by_avgpool2d(model)
    model = replace_activation_by_floor(model, t=16)
    
    model.cuda(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    model.load_state_dict(torch.load('./saved/'+args.id+'.pth'))

    acc = eval_ann(test, model, 0, device=device, rank=rank)
    dist.all_reduce(acc)

    acc/=50000
    if rank == 0:
        print(acc)
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