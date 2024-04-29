#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 13:23:00 2023

@author: emmanuel
"""

# TODO docstring
from sys import path
path.insert(0, "..") # hack to get module `pyEBSD` in scope
import pyEBSD as ebsd
import matplotlib.pyplot as plt
import numpy as np
import math
from numpy import deg2rad, pi







def denoising_pipeline_ctf(noisy_ctf_file,clean_ebsd_file, preprocess=False, denoise=False, denoise_type='tvflow', postprocess=False, postprocess_type='median', identify_isolated_pts=False, l2error=True, plots=True):
    """ A pipeline for preprocessing, denoising, inpainting, postprocessing
    Parameters
    ----------
    noisy_ctf_file : str
        A ctf file containing the noisy orientation data. Should be entered with the .ctf extension
    clean_ebsd_file : str
        A ctf file containing the clean orientation data. Should be entered with the .ctf extension 
    Returns
    -------
    numpy.ndarray
        clean file
        noisy file
        denoising pipeline results
        name of the output file
    Raises
    ------
    ValueError   
    """
    ""
    ebsd_file = deg2rad(ebsd.fileio.read_ctf(noisy_ctf_file)) # Read noisy file

    "Preprocessing step"
    if preprocess==True:
        prepped = ebsd.orient.clean_discontinuities(ebsd_file)
        prepped = ebsd.orient.fill_isolated_with_median(prepped)
    else:
        prepped = ebsd_file
    
    "Denoising step"
    if denoise==True and denoise_type=='tvflow':
        denoised = ebsd.denoise(prepped, weighted=False, beta=0.001, on_quats=False, force_max_iters=False)
    elif denoise==True and denoise_type=='weighted_tvflow':
        denoised = ebsd.denoise(prepped, weighted=True, beta=0.001, on_quats=False, force_max_iters=False)
    elif denoise==True and denoise_type=='median':
        denoised = ebsd.orient.apply_median_filter(prepped)
    else:
        denoised = prepped
        
    "Postprocessing step"
    if postprocess==True and postprocess_type == 'median':
        if identify_isolated_pts==True:
            postprocessed = ebsd.orient.fill_isolated_with_median(denoised)
        else:
            postprocessed = ebsd.orient.apply_median_filter(denoised)
    elif postprocess==True and postprocess_type == 'tvflow':
        if identify_isolated_pts==True:
            postprocessed = ebsd.orient.inpaint_isolated_pts(denoised)
        else:
            postprocessed = ebsd.inpaint(denoised, delta_tolerance=1e-5, on_quats=False)
    else:
        postprocessed = denoised
    
   
    try:
        clean = deg2rad(ebsd.fileio.read_ctf(clean_ebsd_file))
        
    except:
        clean = deg2rad(ebsd.fileio.read_ctf(noisy_ctf_file[:15]))
        
    ebsd_file = deg2rad(ebsd.fileio.read_ctf(noisy_ctf_file)) # Read noisy file    
    if l2error==True:
        noisy_l2 = ebsd.orient.mean_l2_error_per_pixel(ebsd_file, clean)
        denoised_l2 = ebsd.orient.mean_l2_error_per_pixel(postprocessed,clean)
        print('the mean l2 error of the noisy file is ', noisy_l2)
        print('the mean l2 error is ', denoised_l2)
        print('The percentage improvement of the l2 error is',round((noisy_l2-denoised_l2)*100/noisy_l2, 1),'%')
    
    pipeline_name=''
    if preprocess==True:
        pipeline_name+='Preprocessed'
    if denoise==True and denoise_type=='tvflow':
        pipeline_name+='+ebsd_denoised'
    if denoise==True and denoise_type=='median':
        pipeline_name+='+median_denoised'
    if postprocess==True and postprocess_type=='tvflow':
        pipeline_name+='+ebsd_postprocessed'    
    if postprocess==True and postprocess_type=='median':
        pipeline_name+='+median_postprocessed' 
    
    
    if plots==True:
        plt.figure(figsize=(20,7))
        plt.subplot(1,4,1); plt.imshow(range_map(clean, (0,2*pi))); plt.title('clean')
        plt.subplot(1,4,2); plt.imshow(range_map(ebsd_file, (0,2*pi))); plt.title('noisy')
        plt.subplot(1,4,3); plt.imshow(range_map(postprocessed, (0,2*pi))); plt.title(pipeline_name)
        plt.subplot(1,4,4); plt.imshow(np.sum( np.abs(clean - postprocessed), 2 )); plt.title('difference plot')
        
    ebsd.fileio.save_file(noisy_ctf_file+' '+pipeline_name)
    return clean, ebsd_file, postprocessed, noisy_ctf_file+' '+pipeline_name



def denoising_pipeline_mat(noisy_mat_file,clean_mat_file, preprocess=False, denoise=False, denoise_type='tvflow', postprocess=False, postprocess_type='median', identify_isolated_pts=False, l2error=True, plots=True):
    """ A pipeline for preprocessing, denoising, inpainting, postprocessing
    Parameters
    ----------
    noisy_mat_file : str
        A mat file containing the noisy orientation data. Should be entered with the .mat extension
    clean_ebsd_file : str
        A mat file containing the clean orientation data. Should be entered with the .mat extension 
    Returns
    -------
    numpy.ndarray
        clean file
        noisy file
        denoising pipeline results
        name of the output file
    Raises
    ------
    ValueError   
    """
    ebsd_file = deg2rad(ebsd.fileio.read_mat(noisy_mat_file,'noisy')) # Read noisy file

    "Preprocessing step"
    if preprocess==True:
        prepped = ebsd.orient.clean_discontinuities(ebsd_file)
        prepped = ebsd.orient.fill_isolated_with_median(prepped)
    else:
        prepped = ebsd_file
    
    "Denoising step"
    if denoise==True and denoise_type=='tvflow':
        denoised = ebsd.denoise(prepped, weighted=True, beta=0.001, on_quats=False)
    elif denoise==True and denoise_type=='median':
        denoised = ebsd.orient.apply_median_filter(prepped)
    else:
        denoised = prepped
        
    "Postprocessing step"
    if postprocess==True and postprocess_type == 'median':
        if identify_isolated_pts==True:
            postprocessed = ebsd.orient.fill_isolated_with_median(denoised)
        else:
            postprocessed = ebsd.orient.apply_median_filter(denoised)
    elif postprocess==True and postprocess_type == 'tvflow':
        if identify_isolated_pts==True:
            postprocessed = ebsd.orient.inpaint_isolated_pts(denoised)
        else:
            postprocessed = ebsd.inpaint(denoised, delta_tolerance=1e-5, on_quats=False)
    else:
        postprocessed = denoised
        
    try:
        clean = deg2rad(ebsd.fileio.read_mat(clean_mat_file, 'clean'))
        
    except:
        clean = deg2rad(ebsd.fileio.read_mat(noisy_mat_file[:15], 'clean'))
        
    ebsd_file = deg2rad(ebsd.fileio.read_mat(noisy_mat_file,'noisy')) # Read noisy file    
    if l2error==True:
        noisy_l2 = ebsd.orient.mean_l2_error_per_pixel(ebsd_file, clean)
        denoised_l2 = ebsd.orient.mean_l2_error_per_pixel(postprocessed,clean)
        print('the mean l2 error of the noisy file is ', noisy_l2)
        print('the mean l2 error is ', denoised_l2)
        print('The percentage improvement of the l2 error is',round((noisy_l2-denoised_l2)*100/noisy_l2, 1),'%')
    
    pipeline_name=''
    if preprocess==True:
        pipeline_name+='Preprocessed'
    if denoise==True and denoise_type=='tvflow':
        pipeline_name+='+ebsd_denoised'
    if denoise==True and denoise_type=='median':
        pipeline_name+='+median_denoised'
    if postprocess==True and postprocess_type=='tvflow':
        pipeline_name+='+ebsd_postprocessed'    
    if postprocess==True and postprocess_type=='median':
        pipeline_name+='+median_postprocessed' 
    
    
    if plots==True:
        plt.figure(figsize=(20,7))
        plt.subplot(1,4,1); plt.imshow(range_map(clean, (0,2*pi))); plt.title('clean')
        plt.subplot(1,4,2); plt.imshow(range_map(ebsd_file, (0,2*pi))); plt.title('noisy')
        plt.subplot(1,4,3); plt.imshow(range_map(postprocessed, (0,2*pi))); plt.title(pipeline_name)
        plt.subplot(1,4,4); plt.imshow(np.sum( np.abs(clean - postprocessed), 2 )); plt.title('difference plot')
        
    ebsd.fileio.save_file(noisy_mat_file+' '+pipeline_name)
    return clean, ebsd_file, postprocessed, noisy_mat_file+' '+pipeline_name
