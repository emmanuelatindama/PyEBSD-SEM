#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 09:47:19 2023
@author: emmanuel
"""

import matplotlib.pyplot as plt
from numpy import deg2rad, rad2deg, pi, isnan
from sys import path
path.insert(0, "..") # hack to get module `pyEBSD` in scope
import pyEBSD as ebsd
from pyEBSD.misc import range_map

import numpy as np
from scipy.spatial.transform import Rotation as R


# ebsd_data1 = np.load('../EBSDtrainingdata_200by200/1.npy')
# ebsd_data2 = np.load('../EBSDtrainingdata_200by200/2.npy')

test1 = [30, 45, 60]
r1 = R.from_euler('ZXZ', test1, degrees=True)
x1 = r1.as_matrix()
x1 = np.array([1, 0, 0, 0, 0, -1, 0, 1, 0]).reshape((3,3))

# tovec1 = R.from_matrix(x1)
# tovec1 = tovec1.as_euler('ZXZ', degrees=True)
# print('\n the true rotation is', tovec1)

test2 = [30+90, 45, 60]
r2 = R.from_euler('ZXZ', test2, degrees=True)
x2 = r2.as_matrix() # x1.T
x2 = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape((3,3))

# tovec2 = R.from_matrix(x2)
# tovec2 = tovec2.as_euler('ZXZ', degrees=True)
# print('\n the true rotation is', tovec2)

result = x1@x2.T
result = R.from_matrix(result)
print('\n misorientation between these two rotations is:\n\n', result.as_euler('ZXZ', degrees=False))

# print('old misorientation calculation',ebsd.orient.misorientation(np.array(test1),np.array(test2)))
# print('\nnew misorientation calculation',ebsd.orientation.misorientation(x1, x2, out='deg', symmetry_op='cubic'))

# Example usage:
# Define a rotation matrix (e.g., a 90-degree rotation around the z-axis)

# print('\n',ebsd.orientation.axis_angle_to_rotation_matrix([0, 0, -1], np.radians(60)))



# print('the mean orientaion angle of the given Euler matrices is,',ebsd.orientation.mean_misorientation_ang(ebsd_data1, ebsd_data2, symmetry_op = None))
# print('\nthe mean_squared orientaion angle of the given Euler matrices is,',ebsd.orientation.rtmean_sq_misorientation_ang(ebsd_data1, ebsd_data2, symmetry_op = None))






import tensorflow as tf


class rt_mean_sq_GeodesicLoss(tf.keras.losses.Loss):
    def __init__(self, eps=1e-7, min_val=0, max_val=1, reduction='mean'):
        super().__init__()
        self.eps = eps
        #self.reduction = reduction
        self.min_val = min_val
        self.max_val = max_val

    def ang_to_mat(self, x):
        B, H, W, C = x.shape
        
        c_phi = tf.math.cos(x)
        s_phi = tf.math.sin(x)
        
        #R = tf.zeros((B, H, W, 3, 3))
        
        cphi1 = c_phi[:, :, :, 0, None]
        cPhi = c_phi[:, :, :, 1, None]
        cphi2 = c_phi[:, :, :, 2, None]
        sphi1 = s_phi[:, :, :, 0, None]
        sPhi = s_phi[:, :, :, 1, None]
        sphi2 = s_phi[:,:,:,2, None]
        
        out = []
        
        out.append( cphi1*cphi2 - sphi1*cPhi*sphi2)
        out.append( -cphi1*sphi2 - sphi1*cPhi*cphi2)
        out.append( sphi1*sPhi)
        out.append( sphi1*cphi2 + cphi1*cPhi*sphi2)
        out.append( -sphi1*sphi2 + cphi1*cPhi*cphi2)
        out.append( -cphi1*sPhi)
        out.append( sPhi*sphi2)
        out.append( sPhi*cphi2)
        out.append( cPhi)
        
        R = tf.stack(out, axis=4)
        R = tf.reshape(R, (B, H, W, 3, 3))
        
        return R
        
    def call(self, x, y):
        x = self.ang_to_mat(x)
        y = self.ang_to_mat(y)
        R_diffs = tf.linalg.matmul(x, tf.transpose(y, perm=(0, 1, 2, 4, 3)))
        traces = tf.linalg.trace(R_diffs)
        dists = tf.math.acos(tf.clip_by_value((traces - 1)/2, -1+ self.eps, 1- self.eps))
        
        return tf.reduce_mean(dists**2)**.5


loss = rt_mean_sq_GeodesicLoss(eps=1e-20)
# print(loss(x,y))


# clean = np.load('/home/emmanuel/Desktop/EBSD_thesis_codes/clean.npy')
# noisy = np.load('/home/emmanuel/Desktop/EBSD_thesis_codes/noisy.npy')
# predicted = np.load('/home/emmanuel/Desktop/EBSD_thesis_codes/predicted.npy')
# sharp_predicted = np.load('/home/emmanuel/Desktop/EBSD_thesis_codes/sharp_predicted.npy')

# i = 1
# print('noisy misorientation is ', ebsd.orientation.rtmean_sq_misorientation_ang(clean[i], noisy[i], symmetry_op = None))
    
# print('denoised misorientation is ', ebsd.orientation.rtmean_sq_misorientation_ang(clean[i], predicted[i], symmetry_op = None))
    


# plt.imshow(clean[i]/2/pi); plt.show()
# plt.imshow(noisy[i]/2/pi); plt.show()
# plt.imshow(predicted[i]/2/pi); plt.show()
# plt.imshow(predicted[i,:,:,0]/2/pi); plt.show()
# plt.imshow(predicted[i,:,:,1]/2/pi); plt.show()

# plt.figure(figsize=(10,10)); ebsd.ipf.plotIPF(noisy[i]); plt.title("noisy")
# plt.figure(figsize=(10,10)); ebsd.ipf.plotIPF(predicted[i]); plt.title("denoised")
# plt.figure(figsize=(10,10)); ebsd.ipf.plotIPF(clean[i]); plt.title("ground truth")
