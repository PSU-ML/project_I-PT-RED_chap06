from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import math
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import copy as cp
from numpy import linalg
from scipy.stats import gamma

parser = argparse.ArgumentParser(description='anomaly detection')

args = parser.parse_args()

# Detection parameters
NC = 10
THETA = 0.05

# Load in detection statistics
r = np.zeros((NC, NC))
for s in range(NC):
    for t in range(NC):
        if s == t:
            continue
        pert = torch.load('./pert_estimated/pert_{}_{}'.format(s, t))
        pert_norm = torch.norm(pert)
        r[s, t] = 1/pert_norm

# Get the lower-level order statistic p-value
order_pvs = []
for c in range(NC):
    # Fit a Gamma by excluding statistics with target c
    r_null = []
    r_eval = []
    for s in range(NC):
        for t in range(NC):
            if s == t:
                continue
            if r[s, t] == 0:
                continue
            if t == c:
                r_eval.append(r[s, t])
            else:
                r_null.append(r[s, t])
    shape, loc, scale = gamma.fit(r_null)
    # Evaluate the p-values
    order_pvs.append(1 - pow(gamma.cdf(np.max(r_eval), a=shape, loc=loc, scale=scale), NC-1))  # order statistic p-value of the maximum statistic among r_eval

# Get the upper-level order statistic p-value
pv = 1 - pow(1 - np.min(order_pvs), NC)

# Inference
if pv > THETA:
    print("No backdoor attack!")
else:
    print("Backdoor attack detected!")
    t_est = np.argmin(order_pvs)
    s_est = np.argmax(r[:, t_est])
    print("Detected (s, t) pair: ({}, {})".format(s_est, t_est))



