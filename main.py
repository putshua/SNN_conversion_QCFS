import torch.multiprocessing as mp
import argparse
from Models import modelpool
from Preprocess import datapool
from funcs import *
from utils import replace_activation_by_floor, replace_activation_by_neuron, replace_maxpool2d_by_avgpool2d
from ImageNet.train import main_worker
import torch.nn as nn
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('action', default='train', type=str, help='Action: train or test.')
    parser.add_argument('--gpus', default=1, type=int, help='GPU number to use.')
    parser.add_argument('--bs', default=128, type=int, help='Batchsize')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--wd', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--epochs', default=120, type=int, help='Training epochs')
    parser.add_argument('--id', default=None, type=str, help='Model identifier')
    parser.add_argument('--device', default='cuda', type=str, help='cuda or cpu')
    parser.add_argument('--l', default=16, type=int, help='L')
    parser.add_argument('--t', default=16, type=int, help='T')
    parser.add_argument('--mode', type=str, default='ann')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data', type=str, default='cifar100')
    parser.add_argument('--model', type=str, default='vgg16')
    args = parser.parse_args()
    
    seed_all()

    # only ImageNet using multiprocessing,
    if args.gpus > 1:
        if args.data.lower() != 'imagenet':
            AssertionError('Only ImageNet using multiprocessing.')
        mp.spawn(main_worker, nprocs=args.gpus, args=(args.gpus, args))
    else:
        # preparing data
        train, test = datapool(args.data, args.bs)
        # preparing model
        model = modelpool(args.model, args.data)
        model = replace_maxpool2d_by_avgpool2d(model)
        model = replace_activation_by_floor(model, t=args.l)
        criterion = nn.CrossEntropyLoss()
        if args.action == 'train':
            train_ann(train, test, model, args.epochs, args.device, criterion, args.lr, args.wd, args.id)
        elif args.action == 'test' or args.action == 'evaluate':
            model.load_state_dict(torch.load('./saved_models/' + args.id + '.pth'))
            if args.mode == 'snn':
                model = replace_activation_by_neuron(model)
                model.to(args.device)
                acc = eval_snn(test, model, args.device, args.t)
                print('Accuracy: ', acc)
            elif args.mode == 'ann':
                model.to(args.device)
                acc, _ = eval_ann(test, model, criterion, args.device)
                print('Accuracy: {:.4f}'.format(acc))
            else:
                AssertionError('Unrecognized mode')