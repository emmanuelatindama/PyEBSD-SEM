#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:36:53 2023

@author: emmanuel
"""

import matplotlib.pyplot as plt
from numpy import deg2rad, pi, isnan
from sys import path
path.insert(0, "..") # hack to get module `pyEBSD` in scope
import pyEBSD as ebsd
from pyEBSD.misc import range_map

e = deg2rad(ebsd.fileio.read_ctf("/home/emmanuel/Desktop/EBSD_thesis_codes/EBSDctfFiles/missing.ctf", missing_phase=0))
e = e[300:,300:,:]

eshow = e.copy()
eshow[isnan(eshow)] = 2*pi


plt.imshow(eshow/2/pi); plt.title("missing"); plt.show()

u = ebsd.tvflow.inpaint(e, delta_tolerance=1e-5, on_quats=False, force_max_iters=False, max_iters=100)
plt.imshow(range_map(u, (0, pi))); plt.title("inpainted"); plt.show()


u1 = ebsd.orient.clean_discontinuities(u.copy())
plt.imshow(range_map(u1, (0, pi))); plt.title("preprocessed"); plt.show()


u2 = ebsd.orient.fill_isolated_with_median(u1)
plt.imshow(range_map(u2, (0, pi))); plt.title("median_inpainted"); plt.show()


u3 = ebsd.tvflow.denoise(u2, weighted=True, beta=0.0005, on_quats=False)
plt.imshow(range_map(u3, (0, pi))); plt.title("denoised"); plt.show()

u4 = ebsd.orient.apply_median_filter(u3)
plt.imshow(range_map(u4, (0, pi))); plt.title("postprocessed"); plt.show()
