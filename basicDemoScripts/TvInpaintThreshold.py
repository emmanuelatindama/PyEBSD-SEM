#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 09:47:19 2023

@author: emmanuel
"""

import matplotlib.pyplot as plt
from numpy import deg2rad, pi, isnan
from sys import path
path.insert(0, "..") # hack to get module `pyEBSD` in scope
import pyEBSD as ebsd
from pyEBSD.misc import range_map

import numpy as np



clean = np.load('../data_denoise/test_clean.npy'); clean = clean[:5].astype('float32')
noisy = np.load('../data_denoise/test_noisy.npy'); noisy = noisy[:5].astype('float32')


for i in range(10):
    umed = ebsd.orient.apply_median_filter(noisy[i])
    prep = ebsd.orient.fill_isolated_with_median(noisy[i])
    
    plt.figure(figsize=(10,10)); ebsd.ipf.plotIPF(noisy[i]); plt.title("noisy")
    plt.figure(figsize=(10,10)); ebsd.ipf.plotIPF(umed); plt.title("median_denoised")
    # plt.figure(figsize=(10,10)); ebsd.ipf.plotIPF(clean[i]); plt.title("ground truth")
    plt.figure(figsize=(10,10)); ebsd.ipf.plotIPF(prep); plt.title("inpaint_isolated_pixels")
    
    print(f"experiment {i} is complete")
















