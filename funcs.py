import numpy as np
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
import torch
from tqdm import tqdm
from utils import *
from spikingjelly.clock_driven import encoding
from modules import LabelSmoothing
import torch.distributed as dist
import random
import os

def seed_all(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def eval_ann(test_dataloader, model, epoch, device, rank=0):
    epoch_loss = 0
    tot = torch.tensor(0.).cuda(device)
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    model.cuda(device)
    if rank == 0:
        data = tqdm(test_dataloader)
    else:
        data = test_dataloader
    with torch.no_grad():
        for img, label in data:
            img = img.cuda(device, non_blocking=True)
            label = label.cuda(device, non_blocking=True)
            out = model(img)
            loss = loss_fn(out, label)
            epoch_loss += loss.item()
            tot += (label==out.max(1)[1]).sum().data
    return tot

def train_ann(train_dataloader, test_dataloader, model, epochs, device, lr=0.1, save=None, rank=0):
    model.cuda(device)
    para1, para2, para3 = regular_set(model)
    optimizer = torch.optim.SGD([{'params': para1, 'weight_decay':1e-4}, {'params': para2, 'weight_decay':1e-4}, {'params': para3, 'weight_decay':1e-4}], lr=lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = LabelSmoothing(0.1)
    best_acc = 0
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        if rank == 0:
            data = tqdm(train_dataloader)
        else:
            data = train_dataloader
        for idx, (img, label) in enumerate(data):
            img = img.cuda(device, non_blocking=True)
            label = label.cuda(device, non_blocking=True)
            optimizer.zero_grad()
            out = model(img)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        tmp_acc = eval_ann(test_dataloader, model, epoch, device, rank)

        dist.all_reduce(tmp_acc)
        tmp_acc/=50000
        if rank == 0:
            print(f"Epoch {epoch}: Acc: {tmp_acc.item()}")
            if save != None and tmp_acc >= best_acc:
                torch.save(model.state_dict(), './saved/' + save + '.pth')
        best_acc = max(tmp_acc, best_acc)
        scheduler.step()
    return best_acc, model

def eval_snn(test_dataloader, model, device, sim_len=8, rank=0):
    tot = torch.zeros(sim_len).cuda(device)
    model = model.cuda(device)
    model.eval()
    if rank == 0:
        data = tqdm(test_dataloader)
    else:
        data = test_dataloader
    # valuate
    with torch.no_grad():
        for idx, (img, label) in enumerate(data):
            spikes = 0
            img = img.cuda()
            label = label.cuda()
            for t in range(sim_len):
                with torch.cuda.amp.autocast():
                    out = model(img)
                spikes += out
                tot[t] += (label==spikes.max(1)[1]).sum()
            reset_net(model)
    return tot