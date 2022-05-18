from textwrap import fill
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os
from Preprocess.augment import Cutout, CIFAR10Policy

import numpy as np

import tonic
from tonic.slicers import SliceByEventCount
from tonic import SlicedDataset, DiskCachedDataset

# your own data dir
DIR = { 'CIFAR10': '/cluster/scratch/rsrinivasan/datasets',
        'CIFAR100': '/cluster/scratch/rsrinivasan/datasets',
        'ImageNet': 'YOUR_IMAGENET_DIR', 
        'DVSGesture': '/cluster/scratch/rsrinivasan/datasets'
    }

# def GetCifar10(batchsize, attack=False):
#     if attack:
#         trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
#                                   transforms.RandomHorizontalFlip(),
#                                   CIFAR10Policy(),
#                                   transforms.ToTensor(),
#                                   Cutout(n_holes=1, length=16)
#                                   ])
#         trans = transforms.Compose([transforms.ToTensor()])
#     else:
#         trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
#                                   transforms.RandomHorizontalFlip(),
#                                   CIFAR10Policy(),
#                                   transforms.ToTensor(),
#                                   transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#                                   Cutout(n_holes=1, length=8)
#                                   ])
#         trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
#     train_data = datasets.CIFAR10(DIR['CIFAR10'], train=True, transform=trans_t, download=True)
#     test_data = datasets.CIFAR10(DIR['CIFAR10'], train=False, transform=trans, download=True) 
#     train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8, pin_memory=True)
#     test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)
#     return train_dataloader, test_dataloader

def GetCifar10(batchsize, attack=False):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  CIFAR10Policy(),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                  Cutout(n_holes=1, length=16)
                                  ])
    if attack:
        trans = transforms.Compose([transforms.ToTensor()])
    else:
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    train_data = datasets.CIFAR10(DIR['CIFAR10'], train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR10(DIR['CIFAR10'], train=False, transform=trans, download=True) 
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=8)
    return train_dataloader, test_dataloader




def GetCifar100(batchsize):
    trans_t = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]]),
                                  Cutout(n_holes=1, length=16)
                                  ])
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[n/255. for n in [129.3, 124.1, 112.4]], std=[n/255. for n in [68.2,  65.4,  70.4]])])
    train_data = datasets.CIFAR100(DIR['CIFAR100'], train=True, transform=trans_t, download=True)
    test_data = datasets.CIFAR100(DIR['CIFAR100'], train=False, transform=trans, download=True) 
    train_dataloader = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)
    return train_dataloader, test_dataloader

def GetImageNet(batchsize):
    trans_t = transforms.Compose([transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])
    
    trans = transforms.Compose([transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(), 
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])

    train_data = datasets.ImageFolder(root=os.path.join(DIR['ImageNet'], 'train'), transform=trans_t)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_dataloader =DataLoader(train_data, batch_size=batchsize, shuffle=False, num_workers=8, sampler=train_sampler, pin_memory=True)

    test_data = datasets.ImageFolder(root=os.path.join(DIR['ImageNet'], 'val'), transform=trans)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    test_dataloader = DataLoader(test_data, batch_size=batchsize, shuffle=False, num_workers=2, sampler=test_sampler) 
    return train_dataloader, test_dataloader

def GetDVSGesture(batchsize, test_batchsize=4, slicer=SliceByEventCount(event_count=3000), filter_time=10000, time_window=1000, ms_end=300):

    sensor_size = tonic.datasets.DVSGesture.sensor_size
    trans_ann_train = tonic.transforms.Compose([
                            tonic.transforms.Denoise(filter_time=filter_time),
                            tonic.transforms.RandomFlipPolarity(),
                            tonic.transforms.SpatialJitter(sensor_size=sensor_size, clip_outliers=True),
                            tonic.transforms.ToImage(sensor_size=sensor_size),
                            transforms.Lambda(lambda x: torch.from_numpy(x)),
                            Cutout(n_holes=1, length=8)
                        ])

    trans_ann_test = tonic.transforms.Compose([
                            tonic.transforms.Denoise(filter_time=filter_time),
                            tonic.transforms.ToImage(sensor_size=sensor_size),
                            transforms.Lambda(lambda x: torch.from_numpy(x))
                        ])

    trans_snn = tonic.transforms.Compose([
        tonic.transforms.Denoise(filter_time=filter_time),
        tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=time_window),
        transforms.Lambda(lambda x: x[:ms_end, :, :, :]),
    ])

    train_data = tonic.datasets.DVSGesture(save_to=os.path.join(DIR['DVSGesture'], 'train'), train=True, transform=None)
    test_data_ann = tonic.datasets.DVSGesture(save_to=os.path.join(DIR['DVSGesture'], 'test'), train=False, transform=None)
    test_data_snn = tonic.datasets.DVSGesture(save_to=os.path.join(DIR['DVSGesture'], 'test'), train=False, transform=trans_snn)

    sliced_td = SlicedDataset(train_data, slicer=slicer, transform=trans_ann_train, metadata_path=os.path.join(DIR['DVSGesture'], 'metedata/train'))
    sliced_ann = SlicedDataset(test_data_ann, slicer=slicer, transform=trans_ann_test, metadata_path=os.path.join(DIR['DVSGesture'], 'metedata/test'))

    cached_td = DiskCachedDataset(sliced_td, target_transform=None, cache_path=os.path.join(DIR['DVSGesture'], 'cache/train'))
    cached_ann = DiskCachedDataset(sliced_ann, target_transform=None, cache_path=os.path.join(DIR['DVSGesture'], 'cache/test'))

    train_dataloader = DataLoader(cached_td, batch_size=batchsize, shuffle=True, num_workers=8, pin_memory=True)
    test_dataloader_ann = DataLoader(cached_ann, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader_snn = DataLoader(test_data_snn, batch_size=test_batchsize, shuffle=False, num_workers=4, pin_memory=True)

    return train_dataloader, test_dataloader_ann, test_dataloader_snn