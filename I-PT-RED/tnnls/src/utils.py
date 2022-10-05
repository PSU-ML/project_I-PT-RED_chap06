import os
import sys
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
import random
import copy as cp


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def pert_est_class_pair(source, target, model, images, labels, pi=0.9, lr=1e-4, NSTEP=100, verbose=False):
    '''
    :param source: souce class
    :param target: target class
    :param model: model to be insected
    :param images: batch of images for perturbation estimation
    :param labels: the target labels
    :param pi: the target misclassification fraction (default is 0.9)
    :param lr: learning rate (default is 1e-4)
    :param NSTEP: number of steps to terminate (default is 100)
    :param verbose: set True to plot details
    :return:
    '''

    if verbose:
        print("Perturbation estimation for class pair (s, t)".format(source, target))

    # Initialize perturbation
    pert = torch.zeros_like(images[0]).to(device)
    pert.requires_grad = True

    for iter_idx in range(NSTEP):

        # Optimizer: SGD
        optimizer = torch.optim.SGD([pert], lr=lr, momentum=0)

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Get the loss
        images_perturbed = torch.clamp(images + pert, min=0, max=1)
        outputs = model(images_perturbed)
        loss = criterion(outputs, labels)

        # Update perturbation
        model.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Compute misclassification fraction rho
        misclassification = 0
        with torch.no_grad():
            images_perturbed = torch.clamp(images + pert, min=0, max=1)
            outputs = model(images_perturbed)
            _, predicted = outputs.max(1)
            misclassification += predicted.eq(labels).sum().item()
            rho = misclassification / len(labels)

        if verbose:
            print("current misclassification: {}; perturbation norm: {}".format(rho, torch.norm(pert).detach().cpu().numpy()))

        # Stopping criteria
        if rho > pi:
            break

    return pert
