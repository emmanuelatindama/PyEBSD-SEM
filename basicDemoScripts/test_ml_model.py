#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 09:34:08 2023

@author: emmanuel
"""

from sys import path
path.insert(0, "..") # hack to get module `pyEBSD` in scope
import pyEBSD as ebsd
# from numpy import deg2rad
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf



class GeodesicLoss(tf.keras.losses.Loss):
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
        
        return tf.reduce_mean(dists)



# clean = np.load('/home/emmanuel/Desktop/EBSD_thesis_codes/clean.npy')
# noisy = np.load('/home/emmanuel/Desktop/EBSD_thesis_codes/noisy.npy')
# masks = np.load('/home/emmanuel/Desktop/EBSD_thesis_codes/masks.npy')

# autoencoder = load_model('resnethuber_weights.09-0.001.h5', compile=False)
# # autoencoder = load_model('resnetl1_weights.05-0.016.h5', compile=False)


# # autoencoder.summary()
# # decoded_imgs = autoencoder.predict([noisy/2/np.pi, masks])
# decoded_imgs = autoencoder.predict(noisy/2/np.pi)

# i = 1

# plt.figure(figsize=(10,10)); ebsd.ipf.plotIPF(noisy[i]); plt.title("noisy")
# plt.figure(figsize=(10,10)); ebsd.ipf.plotIPF(decoded_imgs[i].reshape((200,200,-1))*2*np.pi); plt.title("denoised")
# plt.figure(figsize=(10,10)); ebsd.ipf.plotIPF(clean[i]); plt.title("ground truth")

# plt.imshow(noisy[i]/2/np.pi); plt.axis('off'); plt.show()
# plt.imshow(decoded_imgs[i]); plt.axis('off'); plt.show()
# plt.imshow(clean[i]/2/np.pi); plt.axis('off'); plt.show()


# # plt.figure(figsize=(15,15));
# # plt.subplot(2,2,1); plt.imshow(noisy[i]/2/np.pi*masks[i]); plt.axis('off')
# # plt.subplot(2,2,2); plt.imshow(decoded_imgs[i]); plt.axis('off')
# # plt.subplot(2,2,3); plt.imshow(clean[i]/2/np.pi); plt.axis('off')
# # plt.subplot(2,2,4); plt.imshow(noisy[i]/2/np.pi); plt.axis('off')


import skimage.io
import PIL

original = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper/im_oneblock/0_Original.npy')*np.pi
original = ebsd.ipf.saveIPF(original); plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper/im_oneblock/original.png', original)

og_pconv = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper/im_oneblock/0_og_pconv.npy')*np.pi
og_pconv = ebsd.ipf.saveIPF(og_pconv); plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper/im_oneblock/og_pconv.png', og_pconv)

ml = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper/im_oneblock/0_ml.npy')*np.pi
ml = ebsd.ipf.saveIPF(ml); plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper/im_oneblock/ml.png', ml)

hybrid = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper/im_oneblock/0_hybrid_00.npy')*np.pi
hybrid = ebsd.ipf.saveIPF(hybrid); plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper/im_oneblock/hybrid.png', hybrid)

damaged = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper/im_oneblock/0_Damaged.npy')*np.pi
damaged1 = ebsd.ipf.saveIPF(damaged); x = plt.imread('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper/im_oneblock/0_Damaged.png')
for i in range(damaged.shape[0]):
    for j in range(damaged.shape[1]):
        if x[i,j,0]>0.98 and x[i,j,1]>0.98 and x[i,j,2]>0.98:
            damaged1[i,j,:]=1
plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper/im_oneblock/damaged.png', damaged1)

criminisi_ssdelta = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper/im_oneblock/0_Criminisi (SSDelta).npy')*np.pi
criminisi_ssdelta = ebsd.ipf.saveIPF(criminisi_ssdelta); plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper/im_oneblock/criminisi_ssdelta.png', criminisi_ssdelta)

criminisi_standard = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper/im_oneblock/0_Criminisi (Standard).npy')*np.pi
criminisi_standard = ebsd.ipf.saveIPF(criminisi_standard); plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper/im_oneblock/criminisi_standard.png', criminisi_standard)

criminisi_ssdelta_eu = np.load('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper/im_oneblock/0_Criminisi (SSDelta, Euclidean (1)).npy')*np.pi
criminisi_ssdelta_eu = ebsd.ipf.saveIPF(criminisi_ssdelta_eu); plt.imsave('/home/emmanuel/Desktop/images_for_IMMI_paper/pconv_paper/im_oneblock/criminisi_ssdelta_eu.png', criminisi_ssdelta_eu)







