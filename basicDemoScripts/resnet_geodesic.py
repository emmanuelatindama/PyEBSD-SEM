#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 20:15:13 2023

@author: emmanuel


plt - for plotting and visualizing data and results
np  - for array manipulation and computation
pd  - for manipulating dataframes
tf  - software library for machine learning and artificial intelligence

tensorflow.keras.datasets - small numpy dataset for testing  your code. It contains the following:

1. boston_housing - housing price regression dataset
2. cifar10        - small image classification dataset...probably has a target variable
3. cifar100       - small images classification dataset...probably has a target variable
4. fashion_mnist  - small images of fashion dataset (reshape to visualize)
5. imdb           - IMDB sentiment classification dataset
6. mnist          - MNIST handwritten digits dataset (images...reshape to visualize) has a target variable
7. reuters        - news data for topic classification (probably has a target vector, but no images)


2.0 : here we will load our own data from a data directory, create a mask over the 
data and then train a UNet inpainting model to reconstruct the data. This was done in 1.0 already ans saved as .npy
so we will just load it directly

This takes the masked-image and mask as inputs, thus the algorithm knows what the masked region is. So it gives better results than the vanilla conv net.
We include a custom loss function (custom loss) that uses only the known region as given by the mask
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Concatenate, MaxPooling2D
from tensorflow.keras.models import Model

from sys import path
path.insert(0, "..")
import pyEBSD as ebsd



class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size, res_conv=False, groups=32):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2 = filters

        self.conv1 = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.norm1 = tf.keras.layers.GroupNormalization(groups=groups)

        self.conv2 = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
        self.norm2 = tf.keras.layers.GroupNormalization(groups=groups)
        
        self.res_conv = tf.keras.layers.Conv2D(filters2, (1, 1)) if res_conv else tf.keras.layers.Identity()


    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.norm1(x, training=training)
        x = tf.nn.silu(x)

        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = tf.nn.silu(x)

        x = x + self.res_conv(input_tensor)
        
        return x


class GeodesicLoss(tf.keras.losses.Loss):
    def __init__(self, eps=1e-7, reduction='mean'):
        super().__init__()
        self.eps = eps
        #self.reduction = reduction
        

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
        '''
        if self.reduction == 'none':
            return dists
        if self.reduction == 'mean':
            return tf.math.reduce_mean(dists)
        if self.reduction == 'sum':
            return tf.math.reduce_sum(dists)
        '''


"Model Definition"
# Build U-Net Inpainting Model
def build_unet_inpainting_model(input_shape):
    input_image = Input(input_shape)
    
    # Encoder
    conv1 = ResnetIdentityBlock((64, 64), 5, res_conv=True)
    conv1_output = conv1.call(input_image) # No downsampling 400x400   ------------------------------     #|
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_output) # Downsampling 200x200   ------------------------------     #|
                                                                                                                                #|
    conv2 = ResnetIdentityBlock((128, 128), 3, res_conv=True)
    conv2_output = conv2.call(pool1) # No downsampling 200x200   ------------------------------     #|
    pool2 =  MaxPooling2D(pool_size=(2, 2))(conv2_output) # Downsampling 100x100   ------------------------------     #|                                                                          #|     #|
    
    conv3 = ResnetIdentityBlock((256,256), 3, res_conv=True)
    conv3_output = conv3.call(pool2)  # No downsampling 100x100   -------------------------    #|     #|  
    pool3 =  MaxPooling2D(pool_size=(2, 2))(conv3_output) # Downsampling 50x50   ------------------------------     #|                                                                           #|     #|
                                                                                                                       #|    #|     #|
    conv4 = ResnetIdentityBlock((512, 512), 3, res_conv=True)
    conv4_output = conv4.call(pool3)  # # No downsampling 50x50                #|    #|     #|
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_output)  # Downsampling 25x25   ----------------------    #|    #|     #|
                                                                                                                 #|    #|    #|     #|
    # Middle                                                                                                     #|    #|    #|     #|
    mid_conv = ResnetIdentityBlock((1024, 2048), 3, res_conv=True)
    mid_conv_output = mid_conv.call(pool4)  # No downsampling 25x25                #|    #|     #|
                                                                                                                 #|    #|    #|     #|
    # Decoder                                                                                                    #|    #|    #|     #|
    up4 = UpSampling2D(size=(2, 2))(mid_conv_output)  # Upsamples to 50x50                                              #|    #|    #|     #|
    deconv4 = ResnetIdentityBlock((1024, 512), 3, res_conv=True)
    deconv4_output = deconv4.call(up4)   #---------------    #|    #|     #|                                             #|    #|    #|     #|
    deconv4 = ResnetIdentityBlock((512,256), 3, res_conv=True)
    deconv4_output = deconv4.call(Concatenate()([deconv4_output, conv4_output]))   #---------------    #|    #|     #|
                                                                                                                       #|    #|     #|
    up3 = UpSampling2D(size=(2, 2))(deconv4_output)  # Upsamples to 100x100                                              #|    #|    #|     #|
    deconv3 = ResnetIdentityBlock((256, 256), 3, res_conv=True)
    deconv3_output = deconv3.call(up3)   #---------------    #|    #|     #|                                             #|    #|    #|     #|
    deconv3 = ResnetIdentityBlock((256, 128), 3, res_conv=True)
    deconv3_output = deconv3.call(Concatenate()([deconv3_output, conv3_output]))   #---------------    #|    #|     #|
                                                                                                                             #|     #|
    up2 = UpSampling2D(size=(2, 2))(deconv3_output)  # Upsamples to 200x200                                              #|    #|    #|     #|
    deconv2 = ResnetIdentityBlock((128, 128), 3, res_conv=True)
    deconv2_output = deconv2.call(up2)   #---------------    #|    #|     #|                                             #|    #|    #|     #|
    deconv2 = ResnetIdentityBlock((128,64), 3, res_conv=True)
    deconv2_output = deconv2.call(Concatenate()([deconv2_output, conv2_output]))   #---------------    #|    #|     #|
                                                                                                                                    #|      
    up1 = UpSampling2D(size=(2, 2))(deconv2_output)  # Upsamples to 200x200                                              #|    #|    #|     #|
    deconv1 = ResnetIdentityBlock((64, 64), 5, res_conv=True)
    deconv1_output = deconv1.call(up1)   #---------------    #|    #|     #|                                             #|    #|    #|     #|
    deconv1 = ResnetIdentityBlock((32,32), 3, res_conv=True)
    deconv1_output = deconv1.call(Concatenate()([deconv1_output, conv1_output]))   #---------------    #|    #|     #|
    
    # Output layer
    output_image = Conv2D(3, 1, padding='same')(deconv1_output)
    
    model = Model(inputs=input_image, outputs=output_image)
    return model

autoencoder = build_unet_inpainting_model((400,400,3))
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss = GeodesicLoss(eps=1e-7), run_eagerly=True, metrics=['accuracy'])
autoencoder.summary()
##=========================================================================================================================================



def plot_loss_accuracy(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf[['loss','accuracy']].plot(ylim=(-.1, max(1, historydf.values.max())))
    training_loss = history.history['loss'][-1]
    training_accuracy = history.history['accuracy'][-1]
    plt.title('Training Loss: %.3f, Training Accuracy: %.3f' % (training_loss, training_accuracy))

    plt.figure(figsize=(8, 6))
    historydf[['val_loss','val_accuracy']].plot(ylim=(-.1, max(1, historydf.values.max())))
    val_loss = history.history['val_loss'][-1]
    val_accuracy = history.history['val_accuracy'][-1]
    plt.title('Validation Loss: %.3f, Validation Accuracy: %.3f' % (val_loss, val_accuracy))


def compare_loss_accuracy(history):
    historydf = pd.DataFrame(history.history, index=history.epoch)
    plt.figure(figsize=(8, 6))
    historydf[['loss','val_loss']].plot()
    training_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]
    plt.title('Training Loss: %.3f, Validation Loss: %.3f' % (training_loss, val_loss))

    plt.figure(figsize=(8, 6))
    historydf[['accuracy','val_accuracy']].plot()
    training_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    plt.title('Training Accuracy: %.3f, Validation Accuracy: %.3f' % (training_acc, val_acc))
    
    



"Fit model"
train_noisy_image = np.load('./data/denoise/train_denoise_noisy.npy')
x_train = np.load('./data/denoise/train_denoise_clean.npy')
"Fit model"
history = autoencoder.fit(train_noisy_image, x_train, epochs=15, shuffle=True, batch_size=32, validation_split = 0.1)
# plot training progress

plot_loss_accuracy(history); plt.show(); plt.savefig('./outputs/lossAccuracyPlot_geodesic.jpg')
compare_loss_accuracy(history); plt.show(); plt.savefig('./outputs/compareLossAccuracyPlot_geodesic.jpg')
autoencoder.save('./outputs/geodesic_model.h5')
decoded_imgs = autoencoder.predict(train_noisy_image[:20])


n = 3
plt.figure(figsize=(10, 6))
for i in range(n):
    # display noisy
    ax = plt.subplot(3, n, i + 1)
    ebsd.ipf.plotIPF((train_noisy_image[i].reshape((400,400,-1)))/2/np.pi) #this needs .numpy() since they are converted to tensors during noise introduction
    plt.title("noisy")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + 1 + n)
    ebsd.ipf.plotIPF(decoded_imgs[i].reshape((400,400,-1))/2/np.pi)
    plt.title("denoised")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display ground truth
    ax = plt.subplot(3, n, i + 1 + 2*n)
    ebsd.ipf.plotIPF(x_train[i].reshape((400,400,-1))/2/np.pi)
    plt.title("ground truth")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig('./outputs/geodesic_testcase.jpg', pad_inches=0.01)