import torch.distributed as dist
import torch.multiprocessing as mp
import argparse
from torch import nn
import torch
from Preprocess.getdataloader import GetImageNet
import os
import random
from Models.VGG import vgg16
from Models.ResNet import resnet34
from funcs import eval_snn, eval_ann, seed_all
from utils import replace_activation_by_floor, replace_activation_by_neuron, replace_maxpool2d_by_avgpool2d, search_fold_and_remove_bn
import numpy as np

def main_tester(rank, gpus, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='gloo', rank=rank, world_size=gpus)
    
    device=f'cuda:{rank}'
    torch.cuda.set_device(device)
    seed_all()

    batchsize = int(args.batchsize / gpus)
    train, test = GetImageNet(batchsize)

    # model preparing
    model = resnet34(num_classes=1000)
    model = replace_maxpool2d_by_avgpool2d(model)
    model = replace_activation_by_floor(model, t=16)
    
    model.load_state_dict(torch.load('./saved/'+args.id+'.pth'))

    model = replace_activation_by_neuron(model)
    search_fold_and_remove_bn(model)
    
    model.cuda(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    acc = eval_snn(test, model, device=device, sim_len=args.t, rank=rank)

    dist.all_reduce(acc)
    acc/=50000
    if rank == 0:
        print(acc)

    dist.destroy_process_group()

def main_anntester(rank, gpus, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group(backend='gloo', rank=rank, world_size=gpus)
    
    device=f'cuda:{rank}'
    torch.cuda.set_device(device)
    seed_all()

    batchsize = int(args.batchsize / gpus)
    train, test = GetImageNet(batchsize)

    # model preparing
    model = resnet34(num_classes=1000)
    model = replace_maxpool2d_by_avgpool2d(model)
    model = replace_activation_by_floor(model, t=args.t)
    
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
    parser.add_argument('--id', default=None, type=str, help='Model identifier')
    parser.add_argument('--t', default=16, type=int, help='Time step')
    parser.add_argument('--mode', default='snn', type=str, help='test ann or snn')

    args = parser.parse_args()

    if args.mode == 'snn':
        mp.spawn(main_tester, nprocs=args.gpus, args=(args.gpus, args))
    else:
        mp.spawn(main_anntester, nprocs=args.gpus, args=(args.gpus, args))