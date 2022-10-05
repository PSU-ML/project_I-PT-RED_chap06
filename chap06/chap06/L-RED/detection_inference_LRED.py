import os
import sys
import time
import argparse
import torch
import numpy as np


parser = argparse.ArgumentParser(description='Reverse engineer backdoor pattern')
args = parser.parse_args()

NC = 10
TC = 1
PI = 0.8

# Load in detection statistics
pert_norm = torch.zeros((NC,))
rho_exp = torch.zeros((NC,))
for c in range(NC):
    pert = torch.load('./pert_estimated_weighted_entropy/pert_{}'.format(c))
    pert_norm[c] = torch.norm(pert)
    # rho_exp[c] = torch.load('./pert_estimated_weighted_entropy/rho_exp_{}'.format(c))
pert_norm = pert_norm.numpy()
# print(pert_norm)
# print(rho_exp)

# pert_norm_effective = pert_norm[np.where(rho_exp >= PI)]
pert_norm_effective = pert_norm

consistency_constant = 1.4826  # if normal distribution
median = np.median(pert_norm_effective)
mad = consistency_constant * np.median(np.abs(pert_norm_effective - median))
min_mad = np.abs(np.min(pert_norm) - median) / mad
t = np.argmin(pert_norm)
print(mad)
print("MAD: {}".format(min_mad))

detection_flag = False
if min_mad <= 2.0:
    print("Inference: No attack!")
else:
    print("Inference: There is attack with mad {}, target class {}".format(min_mad, t))
    detection_flag = True

# Detection inference log
if detection_flag is True:
    if t == TC:
        print("Source class correctly inferred.")
    else:
        print("Source class incorrectly inferred.")

