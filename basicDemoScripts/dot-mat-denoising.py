# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 21:07:57 2021

@author: 13152
"""

from sys import path
path.insert(0, "..") # hack to get module `pyEBSD` in scope
import pyEBSD as ebsd
from numpy import pi, mean, deg2rad, isnan, zeros_like, nan
import matplotlib.pyplot as plt


clean = deg2rad(ebsd.fileio.read_mat('e_clean','clean'))
# clean = deg2rad(ebsd.fileio.read_mat('/home/emmanuel/Desktop/hexagonal_4deg/s1'+'/e_clean','clean'))
noise = deg2rad(ebsd.fileio.read_mat('e_noisy','noisy')); #noise = noise
noise = deg2rad(ebsd.fileio.read_mat('/home/emmanuel/Desktop/hexagonal_4deg/s1'+'/e_noisy','noisy')) 
quad = deg2rad(ebsd.fileio.read_mat('e_q','e_q')); # a .mat file restored using Mtex's half-quadratic filter
# quad = deg2rad(ebsd.fileio.read_mat('/home/emmanuel/Desktop/hexagonal_4deg/s1'+'/e_q','e_q'))


"""denoise"""
prep = ebsd.orient.clean_discontinuities(noise.copy())
prepped = ebsd.orient.fill_isolated_with_median(prep,5)

denoised_wtv = ebsd.denoise(prepped, weighted=True, beta=0.001, on_quats=False, max_iters = 8000)


plt.figure(figsize=(10,10)); plt.imshow(clean/(1*pi)); plt.axis('off');plt.savefig('clean.png',bbox_inches='tight',pad_inches=0)
plt.figure(figsize=(10,10)); plt.imshow(noise/(1*pi)); plt.axis('off');plt.savefig('noisy.png',bbox_inches='tight',pad_inches=0)
plt.figure(figsize=(10,10)); plt.imshow(quad/(1*pi)); plt.axis('off');plt.savefig('quad.png',bbox_inches='tight',pad_inches=0)
plt.figure(figsize=(10,10)); plt.imshow(denoised_wtv/(1*pi)); plt.axis('off');plt.savefig('wTV.png',bbox_inches='tight',pad_inches=0)


print('the l2 error of the noisy image is ', ebsd.orient.mean_l2_error_per_pixel(clean,noise))
print('the l2 error of the half-quadratic denoising is ', ebsd.orient.mean_l2_error_per_pixel(clean,quad))
print('the l2 error of the wtv denoising is ', ebsd.orient.mean_l2_error_per_pixel(clean, denoised_wtv))


"saving output files as ctf"
# ebsd.fileio.save_ang_data_as_ctf('wtv.ctf',denoised_wtv)
# ebsd.fileio.save_ang_data_as_ctf('quad.ctf',quad)
# ebsd.fileio.save_ang_data_as_ctf('noisy.ctf',noise)
# ebsd.fileio.save_ang_data_as_ctf('clean.ctf',clean)

