"""
Before/During training detection of imperceptible backdoors
Objective function finalized:
minimize_v  mean_x_in_Ds -log(pt(x+v)) - mean_x_in_Dt -log(1-pt(x-v))
Author: Zhen Xiang
Date: 4/13/2020
"""

from __future__ import absolute_import
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import math
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import random
import copy as cp
import numpy as np

from src.submodels import Net1, Net2

parser = argparse.ArgumentParser(description='Reverse engineer backdoor pattern')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed()

# Detection parameters
NC = 10
NI = 200
PI = 0.8
NSTEP = 300
TC = 6
batch_size = 200

# Load model
net_pre = Net1()
net_post = Net2()
net_pre = net_pre.to(device)
net_post = net_post.to(device)
criterion = nn.CrossEntropyLoss()
'''
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
'''
net_pre.load_state_dict(torch.load('./contam/model_pre.pth'))
net_post.load_state_dict(torch.load('./contam/model_post.pth'))
net_pre.eval()
net_post.eval()

# Create saving path for results
if not os.path.isdir('pert_estimated_baseline'):
    os.mkdir('pert_estimated_baseline')

# Load clean test images
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)


# Learning rate scheduler
def lr_scheduler(iter_idx):
    lr = 1e-3
    if iter_idx > 250:
        lr *= 1e-3
    elif iter_idx > 200:
        lr *= 1e-2
    elif iter_idx > 150:
        lr *= 1e-1

    return lr


# Pert optimization for each class pair
for s in range(NC):
    # Get the clean images from class sc
    ind = [i for i, label in enumerate(testset.targets) if label == s]
    ind = np.random.choice(ind, NI, False)
    images = None
    for i in ind:
        if images is not None:
            images = torch.cat([images, testset.__getitem__(i)[0].unsqueeze(0)], dim=0)
        else:
            images = testset.__getitem__(i)[0].unsqueeze(0)
    images = images.to(device)

    for t in range(NC):
        if t == s:
            continue
        x = cp.copy(images)
        pert = torch.zeros_like(x[0]).to(device)
        pert.requires_grad = True
        # Pert estimation
        for iter_idx in range(NSTEP):
            optimizer = torch.optim.SGD([pert], lr=lr_scheduler(iter_idx), momentum=0.2)
            sample_ind = np.random.choice(range(NI), batch_size, False)
            x_perturbed = torch.clamp(x[sample_ind] + pert, min=0, max=1)
            labels = t * torch.ones((len(x_perturbed),), dtype=torch.long).to(device)
            outputs = net_post(net_pre(x_perturbed))
            loss = criterion(outputs, labels)
            _, pred = outputs.max(1)
            rho = pred.eq(labels).sum().item() / batch_size
            net_post.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            if iter_idx > NSTEP:
                break
            if rho > PI:
                break
        print(s, t, torch.norm(pert.detach().cpu()), rho)
        torch.save(pert.detach().cpu(), './pert_estimated_baseline/pert_{}_{}'.format(s, t))
        torch.save(torch.norm(pert.detach().cpu()), './pert_estimated_baseline/pert_norm_{}_{}'.format(s, t))
        torch.save(rho, './pert_estimated_baseline/rho_{}_{}'.format(s, t))
