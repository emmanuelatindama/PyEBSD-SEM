#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 14:01:47 2023
@author: emmanuel

This script will run the Criminisi algorithm on all samples in the clean_data directory
Each sample will be ran 10 times, each time with the mask at a different location

The first handful of variables allow easy selection for using
    Onion layering vs Criminisi
    Number of tests per image
    Size of missing region  TODO: Add support for non-rectangular regions
    Set output folder name
    Select similarity measures to use (they are located in criminisi.py)

Output will be a directory which contains one subdirectory per sample in the clean_data directory
Within each subdirectory will be the saved images and log files for each trial run on said sample
(Logs contain error data, location and size of missing region, and options the trial was run with)
Additionally, within the highest directory created will be a file raw_errors.csv, containing all data
related to errors on the trials in one file

#TODO: 
- Add less janky way to resume a run partway through
- Turn this into a function?
"""

from sys import path
path.insert(0, "..") # hack to get module `pyEBSD` in scope

import matplotlib.pyplot as plt
from numpy import deg2rad, pi
import numpy as np
import pyEBSD as ebsd


raw_ebsd = deg2rad(ebsd.fileio.read_ctf("/home/emmanuel/Desktop/EBSD_thesis_codes/EBSDctfFiles/Synthetic_test.ctf"))/2/pi
plt.imshow(raw_ebsd); plt.title('clean'); plt.show()


damage = (slice(15,30), slice(60,75))
damage2 = (slice(65,75), slice(60,70))

dmg_ebsd = np.copy(raw_ebsd)
dmg_ebsd[damage] = 1
dmg_ebsd[damage2] = 1
plt.imshow(dmg_ebsd); plt.title('damaged'); plt.show()

psz = 3

mask =  np.full(raw_ebsd.shape[:2], False)
mask[damage] = True
mask[damage2] = True
plt.imshow(mask); plt.title('mask'); plt.show()


our_result = ebsd.criminisi.inpaint(dmg_ebsd, mask, psz, restrict_search = True, show_movie=False, save_movie=True)
plt.imshow(our_result); plt.title('our result'); plt.show()


crim_result = ebsd.criminisi.inpaint(dmg_ebsd, mask, psz, restrict_search = False)
plt.imshow(crim_result); plt.title('crim result'); plt.show()

