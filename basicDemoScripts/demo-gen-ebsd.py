#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 15:04:24 2023

@author: emmanuel
"""

from sys import path
path.insert(0, "..") # hack to get module `pyEBSD` in scope
import pyEBSD as ebsd
import numpy as np
import matplotlib.pyplot as plt
import random

# ebsd.data_generator.gen_data(datadir='/home/emmanuel/Desktop/EBSD_thesis_codes/EBSDtrainingdata_256by256/', num_files=7574,
#                               pipeline='/home/emmanuel/Desktop/EBSD_thesis_codes/pyEBSD/ipfFolder/clean 256by256.json',
#                               output_file='/home/emmanuel/Desktop/EBSD_thesis_codes/pyEBSD/ipfFolder/clean 256by256 temp.csv',
#                               path_to_dream3d='/home/emmanuel/Desktop/EBSD_thesis_codes/pyEBSD/ipfFolder/DREAM3D/bin', start_num=22427)




# clean = ebsd.data_generator.create_training_data(path_to_ebsdmaps='/home/emmanuel/Desktop/EBSD_thesis_codes/EBSDtrainingdata_256by256', save_path='/home/emmanuel/Desktop/EBSD_thesis_codes/EBSDtrainingdata_256by256/data_denoise/clean')

clean = np.load('../EBSDtrainingdata_256by256/data_denoise/clean.npy')
# noisy = np.empty_like(clean).astype('float16')
# masks = np.empty_like(clean).astype('int8')

noisy = clean

for i in range(clean.shape[0]):
    noisy1 = ebsd.data_generator.add_ebsd_noise(clean[i], np.random.randint(2,7), np.random.uniform(0.01,0.05)) # adding 2 - 6 deg dvlp noise in the interior 1 - 5% impulse noise in the entire image
    if np.random.randint(0,2) == 1: # 1 in 3 chance of happening
        noisy2 = ebsd.data_generator.add_ebsd_noise(clean[i], 2, np.random.uniform(0.2,0.25)) # adding 2 deg dvlp noise in the interior 20 - 25% impulse noise on the boundary
        mask = ebsd.data_generator.damage_image(np.ones_like(clean[i]), clean[i].astype('float32'), InteriorProportion = 0, EdgeProportion=np.random.uniform(0.65,0.85), EdgeThickness=1, Width=1) # mask for adding impulse noise near grain boundaries
        noisy1[np.isnan(mask)]=noisy2[np.isnan(mask)] # mask for adding impulse noise near edges
        
        # adding bands of high and medium noise
        if np.random.randint(0,25) == 1: # 1 in 50 chance of happening
            high_int = np.random.randint(15,25) # width of the band with higher noise
            med_int = np.random.randint(35,45) # width of the band with medium noise
            
            noisy3 = ebsd.data_generator.add_ebsd_noise(clean[i], 2, .1)
            noisy4 = ebsd.data_generator.add_ebsd_noise(clean[i], 2, .15)
            
            # horizontal bands
            noisy1[256-high_int:,:] = noisy4[256-high_int:,:]
            noisy1[256-med_int:256-high_int,:] = noisy3[256-med_int:256-high_int,:]
            
            if np.random.randint(0,2) == 1: # 1 in 100 chance of happening
                # vertical bands
                noisy1[:,:high_int] = noisy4[:,:high_int]
                noisy1[:,high_int:med_int] = noisy3[:,high_int:med_int]
    
    noisy[i] = noisy1
#     masks[i] = 1-np.isnan(ebsd.data_generator.damage_image(np.ones_like(clean[i]), clean[i].astype('float32'), InteriorProportion = np.random.uniform(0,.05), EdgeProportion=np.random.uniform(.15,.4), EdgeThickness=random.choice([1,3,1,1,1]), Width=random.choice([1,1,1,3,1,1,1]))).astype('int8') # mask for adding impulse noise near grain boundaries
    if i%100 ==0: print(f'noisy data  and mask {i} is stored')        



np.save('../EBSDtrainingdata_256by256/data_denoise/noisy', noisy)
# np.save('../EBSDtrainingdata_256by256/data_inpaint/masks', masks)


# # noisy = np.load('../EBSDtrainingdata_256by256/data_denoise/noisy.npy'); noisy = noisy[:10]
# # masks = np.load('../EBSDtrainingdata_256by256/data_inpaint/masks.npy'); masks = masks[:10]

# plt.figure(figsize=(10,10)); ebsd.ipf.plotIPF(noisy[0]*masks[0]); plt.title("noisy")
# plt.figure(figsize=(10,10)); ebsd.ipf.plotIPF(noisy[1]*masks[1]); plt.title("noisy")
# plt.figure(figsize=(10,10)); ebsd.ipf.plotIPF(noisy[2]*masks[2]); plt.title("noisy")
# plt.figure(figsize=(10,10)); ebsd.ipf.plotIPF(noisy[3]*masks[3]); plt.title("noisy")
# plt.figure(figsize=(10,10)); ebsd.ipf.plotIPF(noisy[4]*masks[4]); plt.title("noisy")
# plt.figure(figsize=(10,10)); ebsd.ipf.plotIPF(noisy[5]*masks[5]); plt.title("noisy")
# plt.figure(figsize=(10,10)); ebsd.ipf.plotIPF(noisy[6]*masks[6]); plt.title("noisy")



