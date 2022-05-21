import numpy as np
from torch import nn
import torch
from tqdm import tqdm
from utils import *
import torch.optim as optim
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
    torch.backends.cudnn.deterministic = True

def eval_ann(test_dataloader, model, loss_fn, device, rank=0):
    epoch_loss = 0
    tot = torch.tensor(0.).cuda(device)
    model.eval()
    model.cuda(device)
    length = 0
    with torch.no_grad():
        for img, label in test_dataloader:
            img = img.cuda(device)
            label = label.cuda(device)
            out = model(img)
            loss = loss_fn(out, label)
            epoch_loss += loss.item()
            length += len(label)    
            tot += (label==out.max(1)[1]).sum().data
    return tot/length, epoch_loss/length

def train_ann(train_dataloader, test_dataloader, model, epochs, device, loss_fn, lr=0.1, wd=1e-4, save=None, parallel=False, rank=0):
    print('Training Model')
    model.cuda(device)
    # para1, para2, para3 = regular_set(model)
    # optimizer = torch.optim.SGD([
    #                             {'params': para1, 'weight_decay': wd}, 
    #                             {'params': para2, 'weight_decay': wd}, 
    #                             {'params': para3, 'weight_decay': wd}
    #                             ],
    #                             lr=lr, 
    #                             momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0
    for ct, epoch in enumerate(range(epochs)):
        print('Training Epoch: ', ct)
        epoch_loss = 0
        length = 0
        model.train()
        print(len(train_dataloader))
        for ct, (img, label) in enumerate(train_dataloader):
            img = img.cuda(device)
            label = label.cuda(device)
            optimizer.zero_grad()
            # print('Image shape:', img.shape)
            out = model(img)
            # print(out.shape)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            # print('Computed loss')
            epoch_loss += loss.item()
            length += len(label)
            print(f'{ct} out of {len(train_dataloader)} batches')
            # if ct/len(train_dataloader)*100 % 5 == 0: 
            #     print(f'----- \n Finished all batches {ct/len(train_dataloader)*100} percent')
        tmp_acc, val_loss = eval_ann(test_dataloader, model, loss_fn, device, rank)
        if parallel:
            dist.all_reduce(tmp_acc)
        print('Epoch {} -> Val_loss: {}, Acc: {}'.format(epoch, val_loss, tmp_acc), flush=True)
        if rank == 0 and save != None and tmp_acc >= best_acc:
            torch.save(model.state_dict(), './saved_models/' + save + '.pth')
        best_acc = max(tmp_acc, best_acc)
        print('best_acc: ', best_acc)
        scheduler.step()
    return best_acc, model

def eval_snn(test_dataloader, model, device, sim_len=8, rank=0):
    tot = torch.zeros(sim_len).cuda(device)
    length = 0
    model = model.cuda(device)
    model.eval()
    # valuate
    with torch.no_grad():
        for idx, (img, label) in enumerate(tqdm(test_dataloader)):
            spikes = 0
            length += len(label)
            img = img.cuda()
            label = label.cuda()
            for t in range(sim_len):
                out = model(img)
                spikes += out
                tot[t] += (label==spikes.max(1)[1]).sum()
            reset_net(model)
    return tot/length