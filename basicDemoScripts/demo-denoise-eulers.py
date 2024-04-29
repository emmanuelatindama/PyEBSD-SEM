#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:36:53 2023

@author: emmanuel
"""


from sys import path
path.insert(0, "..") # hack to get module `pyEBSD` in scope
import pyEBSD as ebsd
from numpy import deg2rad,pi
import matplotlib.pyplot as plt

"""Read noisy ctf file and clean ctf file and restore for comparison
   If clean file is not available, use noisy file as both.
   In that case larger l2 errors indicate better restoration.
"""
clean, noisy, preprocessed,filename = ebsd.orient.denoising_pipeline_ctf('/home/emmanuel/Desktop/EBSD_thesis_codes/EBSDctfFiles/Synthetic_test_noisy.ctf', '/home/emmanuel/Desktop/EBSD_thesis_codes/EBSDctfFiles/Synthetic_test.ctf', preprocess=True, denoise=True, denoise_type='tvflow', postprocess=False, l2error=True, plots=True)


"""
   To use a different weight function for the edge map other than the one generated from 
   our literature, you may compute and input the array as
   Below, we use the result of our tv flow as weights for weighted tv flow denoising.
"""
e = deg2rad(ebsd.fileio.read_ctf("/home/emmanuel/Desktop/EBSD_thesis_codes/EBSDctfFiles/Synthetic_test_noisy.ctf"))

u = ebsd.tvflow.denoise(e, weighted=False, beta=0.05, on_quats=False, weight_array=ebsd.misc.weight_from_TV_solution( noisy ))
plt.figure();
plt.imshow(u/2/pi);
plt.show()

