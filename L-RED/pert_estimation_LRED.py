"""
Post-training detection of imperceptible backdoors
Pertrubation estimation problem:
minimize_{v, w}     sum_{s != t} w_s / |Ds| sum_{x_in_Ds} -log(pt(x+v)) - sum_{s != t} w_s log(w_s)
subject to          sum_{s != t} w_s = 1;   w_s >= 0 for all s
Author: Zhen Xiang
Date: 8/30/2020
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
import time

from src.resnet import ResNet18

parser = argparse.ArgumentParser(description='Reverse engineer backdoor pattern')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_time = time.time()
random.seed()

# Detection parameters
NC = 10
NI = 6    # No. clean images (for detection) per class
NSTEP = 200
PI = 0.8
LAMBDA = 0.5
LR_MULTI = 1.5  # Increment multiplier of learning rate

# Load model for inspection
model = ResNet18()
model = model.to(device)
if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
model_path = './model/model.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

# Loss function
criterion = nn.CrossEntropyLoss()

# Create saving path for results
if not os.path.isdir('pert_estimated_weighted_entropy'):
    os.mkdir('pert_estimated_weighted_entropy')

# Load clean test images
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

# Get the subset of test set that is correctly classified (could be ignored if the number of images used for detection is large)
indicator = None
total = 0
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if indicator is None:
            indicator = predicted.eq(targets)
        else:
            indicator = torch.cat((indicator, predicted.eq(targets)))
ind_incorrect = [i for i, element in enumerate(indicator) if element == False]
testset.data = np.delete(testset.data, ind_incorrect, axis=0)
testset.targets = np.delete(testset.targets, ind_incorrect, axis=0)

# Create a subset of images for detection from the clean test set
detectionset = cp.copy(testset)
ind_all = None
for c in range(NC):
    ind = [i for i, label in enumerate(detectionset.targets) if label == c]
    ind = np.random.choice(ind, NI, False)
    if ind_all is None:
        ind_all = ind
    else:
        ind_all = np.concatenate((ind_all, ind))
detectionset.data = detectionset.data[ind_all]
detectionset.targets = np.asarray(detectionset.targets)     # Convert from list to np
detectionset.targets = detectionset.targets[ind_all]        # Get the elements based on ind_all
detectionset.targets = detectionset.targets.tolist()        # Convert np back to list

# Perform pattern estimation for each target class
for t in range(NC):

    # Initialize perturbation
    pert = torch.zeros(detectionset.__getitem__(0)[0].size()).to(device)
    pert.requires_grad = True

    # Initialize the weights
    w = torch.ones((NC,))
    w[t] = 0
    w /= (NC - 1)
    w = w.to(device)

    # Initialize the misclassification for each s to the putative t
    rho = torch.zeros((NC,))
    rho_exp = torch.zeros((1,)).to(device)
    rho_exp_pre = rho_exp
    rho_exp_pre = rho_exp_pre.to(device)

    # Joint estimation of perturbation and weights
    for iter_idx in range(NSTEP):

        # Schedule the learning rate
        lr = 5e-4
        if rho_exp_pre - rho_exp <= 0.005:
            lr *= LR_MULTI
        elif rho_exp - rho_exp_pre > 0.05:
            lr /= LR_MULTI

        # Optimizer: SGD
        optimizer = torch.optim.SGD([pert], lr=lr, momentum=0)

        # Fix the weights, get the loss from each source class
        loss_unweighted = torch.zeros((NC,)).to(device)
        for s in range(NC):
            if s == t:
                continue

            # Get the subset images labeled to s
            sourceset = cp.copy(detectionset)
            ind = [i for i, label in enumerate(sourceset.targets) if label == s]
            sourceset.data = sourceset.data[ind]
            sourceset.targets = [t] * len(sourceset.data)   # Label them to t for pert estimation

            # Get a mini-batch of images from s
            sourcesetloader = torch.utils.data.DataLoader(sourceset, batch_size=NI, shuffle=True, num_workers=1)
            batch_idx, (images, labels) = list(enumerate(sourcesetloader))[0]
            images, labels = images.to(device), labels.to(device)

            # Add the perturbation and feed into classifier
            images_perturbed = torch.clamp(images + pert, min=0, max=1)

            # Loss for class s
            outputs = model(images_perturbed)
            loss_unweighted[s] = criterion(outputs, labels)   # Already averaged, it is the "per-image" loss
        loss_sum = torch.dot(w, loss_unweighted)

        # Update the perturbation
        model.zero_grad()
        loss_sum.backward(retain_graph=True)
        optimizer.step()

        # Update rho
        for s in range(NC):
            if s == t:
                continue

            # Get the subset images labeled to s
            sourceset = cp.copy(detectionset)
            ind = [i for i, label in enumerate(sourceset.targets) if label == s]
            sourceset.data = sourceset.data[ind]
            sourceset.targets = [t] * len(sourceset.data)  # Label them to t for pert estimation
            sourcesetloader = torch.utils.data.DataLoader(sourceset, batch_size=NI, shuffle=True, num_workers=1)

            # Get rho for class s
            rho_class = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(sourcesetloader):
                    images, labels = images.to(device), labels.to(device)
                    images_perturbed = torch.clamp(images + pert, min=0, max=1)
                    outputs = model(images_perturbed)
                    _, predicted = outputs.max(1)
                    rho_class += predicted.eq(labels).sum().item()
                    total += len(labels)
            rho[s] = rho_class / total

        # Get expected rho
        rho_exp_pre = rho_exp
        rho_exp = torch.dot(w, rho.to(device))
        # print(iter_idx, rho_exp, torch.norm(pert))
        # print(rho)

        # Stopping critera

        # Criteria 1: expectation of rho is greater than PI
        if rho_exp > PI:
            break

        # Norm criteria:
        if torch.norm(pert) > 6:
            break

        # Update temperature
        if rho_exp > 0:
            T = -torch.log(rho_exp) * LAMBDA
        # Could also set T to be extemely large
        # T = 1e4

        # Update weight w (if T != inf)
        rho = rho.to(device)
        if rho_exp > 0:
            denom = (torch.sum(torch.exp(rho / T)) - torch.exp(rho[t] / T))
            if denom > 0:
                for s in range(NC):
                    if s == t:
                        continue
                    w[s] = torch.exp(rho[s] / T) / denom
            else:
                print('error')
        elif rho_exp == 0.0:
            w = torch.ones((NC,))
            w[t] = 0
            w /= (NC - 1)
            w = w.to(device)
        else:
            print('error')

    print(t, rho_exp, torch.norm(pert))
    print(w)
    torch.save(pert.detach().cpu(), './pert_estimated_weighted_entropy/pert_{}'.format(t))

print("--- %s seconds ---" % (time.time() - start_time))
torch.save((time.time() - start_time), './pert_estimated_weighted_entropy/time')
