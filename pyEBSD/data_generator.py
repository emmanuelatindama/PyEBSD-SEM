#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 11:03:09 2022

@author: emmanuel
"""

import subprocess
from shutil import move
import numpy as np
from scipy.ndimage import convolve
import platform
import os
import cv2


def gen_data(datadir: str, num_files: int, pipeline: str, 
        output_file: str, path_to_dream3d: bool='./',
        start_num: int=1, as_numpy: bool=True):
    '''
    Runs a Dream3d pipeline multiple times automatically, naming the output files sequentially

    ARGS:
    -----
    str: datadir - Path to the directory we want to save the outputs to
    int: num_files - The number of times to run the pipeline
    str: pipeline - Path to the .json file containing the pipeline
    str: output_file - The path to the output file from the pipeline 
    str: path_to_dream3d - Path to the Dream3d folder containing PipelineRunner
    int: start_num - The number to start with when naming the output files
    bool: as_numpy - If true, output data as .npy files.  Else, output as .csv files
                     (NOTE: Assumes pipeline output is csv.  Also normalizes input to be in [0,2*pi])

    RETURNS:
    --------
    0 if sucessful
    '''
    # where is this file?
    #This bit just makes it so that cmd windows don't keep popping
    #up every time we loop on Windows machines
    #(Not sure if MacOS and Linux need different solutions
    #or if they don't have this problem in the first place)
    if platform.system() == 'Windows':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
    else:
        startupinfo = None
    if datadir[-1] != '/':
        datadir = datadir+'/'
    if path_to_dream3d[-1] != '/':
        path_to_dream3d = path_to_dream3d+'/'
        
    for i in range(start_num, num_files+start_num):
        subprocess.run([path_to_dream3d+'PipelineRunner', '-p', pipeline], startupinfo=startupinfo)
        if as_numpy:
            im = np.genfromtxt(output_file, skip_header=1, delimiter=',', usecols=(1,2,3))
            imshape = int(np.sqrt(len(im)))
            im = im.reshape(imshape, imshape, 3)
            np.save(datadir+'{}.npy'.format(i), im)
        else:
            move(output_file, datadir+'{}.csv'.format(i))
        print('Completed file {} of {}'.format(i,num_files+start_num))

    return 0

# if __name__=='__main__':
#     gen_data(datadir='/home/emmanuel/Desktop/EBSD_thesis_codes/EBSDtrainingdata/', num_files=3, pipeline='/home/emmanuel/Desktop/EBSD_thesis_codes/pyEBSD/ipfFolder/clean 400by400.json', output_file='/home/emmanuel/Desktop/EBSD_thesis_codes/pyEBSD/ipfFolder/clean 400by400 temp.csv', path_to_dream3d='/home/emmanuel/Desktop/EBSD_thesis_codes/pyEBSD/ipfFolder/DREAM3D/bin', start_num=12508)



def create_training_data(path_to_ebsdmaps='/home/emmanuel/Desktop/EBSD_thesis_codes/EBSDtrainingdata_400by400', save_path=None, loop:bool=False):
    # e.g save_path = '/home/emmanuel/Desktop/EBSD_thesis_codes/ebsd_400_clean'
    """Creates training data from ebsd maps in euler angles.
    The ebsd file must be in scaled 0 to 2pi (max). This is for creating training data for UNet model
    Parameters
    ----------
    path_to_ebsdmaps : str
        Path to where the .npy files are stored
    save_path: str
        Path to where the created training data should be stored
    loop: 
        Whether you want loop through each ebsd map again. Default is False
    Returns
    -------
    the name of the ctf file of the noisy data generated
    """
    dataList = []
    for img in os.listdir(path_to_ebsdmaps):
        try:
            ebsd_map = np.load(os.path.join(path_to_ebsdmaps, img))
        except Exception:
            continue
        dataList.append(ebsd_map.astype(np.float16))
        if loop:
            dataList.append(ebsd_map.astype(np.float16))
    if save_path:
        np.save(save_path, np.stack(dataList, axis=0))
    else:
        np.save('/home/emmanuel/Desktop/EBSD_thesis_codes/ebsd_400_clean', np.stack(dataList, axis=0))

    return np.stack(dataList, axis=0) # we stack in axis=0 so that the first index of shape gives us the training data size
    

    



def add_ebsd_noise(ebsd_data, std_dev:int=4, probability:float=0.05, discontinuity=False):
    """Adds gaussian noise to ebsd map in euler angles.
    The ebsd file must be in scaled 0 to 2pi (max). This is for creating training data for UNet model
    Parameters
    ----------
    ebsd_ : numpy.ndarray
        Orientation data in Euler angles
    std_dev: int
        Standard deviation of the noise in degree
    Returns
    -------
    the name of the ctf file of the noisy data generated
    """
   
    if np.max(ebsd_data)>1 and np.max(ebsd_data) <= 2*np.pi+.00001:
        ebsd_data = np.rad2deg(ebsd_data)
    elif np.max(ebsd_data)<1:
        print('data is currently scaled to 1, and should be scaled from 0 to 2*pi otherwise you may have incorrect results' )
        ebsd_data = np.rad2deg(ebsd_data)
    elif np.max(ebsd_data)> 2*np.pi+ 0.00001:
        print(f"the max value is {np.max(ebsd_data)}\n")
        print('data is currently scaled to greater than 2*pi, and should be scaled from 0 to 2*pi otherwise you may have incorrect results')
        ebsd_data  = ebsd_data 
    max_0 = np.max(ebsd_data[:,:,0]); max_1 = np.max(ebsd_data[:,:,1]); max_2 = np.max(ebsd_data[:,:,2])
    
    if type(std_dev)!=int:
        raise Exception("std_dev of noise should be an integer (in degrees)")
    if std_dev<=0:
        print('no noise added, since std_dev<=0')
        return np.deg2rad(ebsd_data).astype(np.float16)
    else:
        ebsd_data = ebsd_data + np.random.normal(0, np.random.randint(np.ceil(std_dev/4),std_dev), ebsd_data.shape)
        
    # add noise using modular arithmetic
    if discontinuity==True:
        for channel in range(ebsd_data.shape[2]):
            if np.max(ebsd_data[:,:,channel])<=max_1:
                ebsd_data[:,:,channel] = np.clip(ebsd_data[:,:,channel],0,max_1-1) #max_1-1 = 179 degrees
            else:
                ebsd_data[:,:,channel] = np.mod(ebsd_data[:,:,channel],max_0-1)
    else:
        for channel in range(ebsd_data.shape[2]):
            if np.max(ebsd_data[:,:,channel])<=max_1:
                ebsd_data[:,:,channel] = np.clip(ebsd_data[:,:,channel],0,max_1-1) #max_1-1 = 179 degrees
            else:
                ebsd_data[:,:,channel] = np.clip(ebsd_data[:,:,channel],0,max_0-1) #max_0-1 = 359 degrees
    
    """Create impulse noise"""
    if probability>0:
        impulse = np.random.uniform(size=(ebsd_data.shape[0],ebsd_data.shape[1]))
        for i in range(impulse.shape[0]):
            for j in range(impulse.shape[1]):
                if impulse[i,j]<=probability:
                    ebsd_data[i,j]=np.array([np.random.uniform(0,max_0), np.random.uniform(0,max_1),np.random.uniform(0,max_2)])
    
    return np.deg2rad(ebsd_data).astype(np.float16)

# noisy_data = np.load('/home/emmanuel/Desktop/EBSD_thesis_codes/ebsd_400_clean.npy')
# for i in range(noisy_data.shape[0]):
#     noisy_data[i] = add_ebsd_noise(noisy_data[i], std_dev=6)
# np.save('/home/emmanuel/Desktop/EBSD_thesis_codes/ebsd_400_noisy', noisy_data)




def damage_image(im_to_damage, im_clean, **kwargs):
    """Creates masked images for a given N dimensional array of images
       Parameters
       ----------
        im_to_damage : np.ndarray
            is an N dimensional numpy array of the image(s) you wish to mask. The dimensions must be the same as the im_clean
            You may use np.ones_like(im_clean) if you want a mask to be returned, otherwise, put im_clean if you want the same image to be damaged
        im_to_damage : np.ndarray
            is an N dimensional numpy array of the image(s) you wish to use as reference for damaging. The dimensions must be the same as im_to_damage
        *kwargs
            EdgeProportion: specifies the proportion of edges to be damaged. Default = 1 (100%)
            InteriorProportion: Specifies the proportion of the interior regions to be damaged. Default = 0.1 (10%)
            EdgeThickness: specifies number of pixels including the edge to be damaged. Must be odd. Default = 1
            Width: specifies width of pixels including the edge to be damaged. Must be odd. Default = 1
            eps: threshold for selecting what is considered an edge. Default is 1e-4
       Returns
       -------
       a masked image
       Note: if input image is noisy, detecting edges may be extremely difficult. Hence you may have to tune the eps parameter
    """
    # Parse options
    edge_proportion = kwargs.get('EdgeProportion', 1); validate_edge_proportion(edge_proportion)
    interior_proportion = kwargs.get('InteriorProportion', 0.1); validate_interior_proportion(interior_proportion)
    edge_thickness = kwargs.get('EdgeThickness', 1); validate_edge_thickness(edge_thickness)
    width = kwargs.get('Width', 1); validate_width(width)
    eps = kwargs.get('eps', 1e-4)
    
    # Check if the input images have the same size and dimensionality
    if im_to_damage.shape != im_clean.shape:
        raise ValueError('im_to_damage and im_clean must have the same size and dimensionality.')

    M, N, nchans = im_clean.shape

    # Get edges
    edges = np.abs(convolve(im_clean[:, :, 0], np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))) > eps
    edges = add_border(edges, edge_thickness // 2)

    # Damage image
    randvals = np.random.rand(M, N)

    # Select edge pixels
    nanmask_edges = (randvals <= edge_proportion) & edges
    nanmask_edges = add_border(nanmask_edges, width // 2)

    # Select interior pixels
    nanmask_interior = (randvals <= interior_proportion) & ~edges

    # Get final mask
    nanmask = nanmask_edges | nanmask_interior
    nanmask = np.repeat(nanmask[:, :, np.newaxis], nchans, axis=2)

    # Destroy pixels
    im_to_damage[nanmask] = np.nan

    return im_to_damage

def validate_interior_proportion(interior_proportion):
    if not isinstance(interior_proportion, (int, float)):
        raise ValueError('InteriorProportion must be numeric.')
    if not 0 <= interior_proportion <= 1:
        raise ValueError('InteriorProportion must be in the range [0, 1].')

def validate_edge_proportion(edge_proportion):
    if not isinstance(edge_proportion, (int, float)):
        raise ValueError('EdgeProportion must be numeric.')
    if not 0 <= edge_proportion <= 1:
        raise ValueError('EdgeProportion must be in the range [0, 1].')

def validate_edge_thickness(edge_thickness):
    if not isinstance(edge_thickness, int):
        raise ValueError('EdgeThickness must be numeric.')
    if edge_thickness <= 0 or edge_thickness % 2 == 0:
        raise ValueError('EdgeThickness must be a positive odd integer.')

def validate_width(width):
    if not isinstance(width, int):
        raise ValueError('Width must be numeric.')
    if width <= 0 or width % 2 == 0:
        raise ValueError('Width must be a positive odd integer.')

def add_border(im, border_thickness):
    for _ in range(border_thickness):
        im = np.logical_or(im, np.roll(im, 1, axis=0))
        im = np.logical_or(im, np.roll(im, -1, axis=0))
        im = np.logical_or(im, np.roll(im, 1, axis=1))
        im = np.logical_or(im, np.roll(im, -1, axis=1))
    return im




def _generate_mask(data_arr, size:int=6):
    """Generates a random irregular mask with lines, circles and elipses"""
    height, width, channels = data_arr.shape
    img = np.zeros((height, width, channels), np.uint8)

    # Set size scale
    if not size:
        size = int((width + height) * 0.005)
    if width < 100 or height < 100:
        raise Exception("Width and Height of mask must be at least 100!")
    
    # Draw random lines
    for _ in range(np.random.randint(1, 10)):
        x1, x2 = np.random.randint(1, width), np.random.randint(1, width)
        y1, y2 = np.random.randint(1, height), np.random.randint(1, height)
        thickness = np.random.randint(1, size)
        cv2.line(img,(x1,y1),(x2,y2),(1,1,1),thickness)
        
    # Draw random circles
    for _ in range(np.random.randint(1, 20)):
        x1, y1 = np.random.randint(1, width), np.random.randint(1, height)
        radius = np.random.randint(2, size)
        cv2.circle(img,(x1,y1),radius,(1,1,1), -1)
        
    # Draw random ellipses
    for _ in range(np.random.randint(1, 20)):
        x1, y1 = np.random.randint(1, width), np.random.randint(1, height)
        s1, s2 = np.random.randint(1, width), np.random.randint(1, height)
        a1, a2, a3 = np.random.randint(3, 180), np.random.randint(3, 180), np.random.randint(3, 180)
        thickness = np.random.randint(1, size)
        cv2.ellipse(img, (x1,y1), (s1,s2), a1, a2, a3,(1,1,1), thickness)
    return 1-img




def generate_rectangular_mask(data_arr:np.ndarray, rect_patch_dim:[int,int], masked_reg_val:int = 1):
    """Creates masked images for a given N dimensional array of images
       Parameters
       ----------
        data_arr : np.ndarray
            is an N dimensional numpy array of the image(s) you wish to mask
        rect_patch_dim : [int,int]
            is a list containing exactly 2 integers for the row(first index) and col(second index) length of the rectagle
       # TODO  center = None
            randomizes center point of the masked region by default. user may enter a fixed center to be used for all the images
       Returns
       -------
       a masked image with the rectangle randomly centered or fixed
                The rectangle dimension are as specified in the inputs
    """
    if masked_reg_val != 0 | masked_reg_val !=1:
        raise Exception('InputError: masked region value must be either 0(min) or 1(max) depending on the design of your algorithm')
    
    
    # Choose a random patch and it corresponding unmask index.
    center = [np.random.randint(rect_patch_dim[1]//2, data_arr.shape[0]-1-rect_patch_dim[0]//2), np.random.randint(rect_patch_dim[1]//2, data_arr.shape[1]-1-rect_patch_dim[1]//2)]
    col_min = center[0] - rect_patch_dim[1]//2
    col_max = center[0] + rect_patch_dim[1]//2
    row_min = center[1] - rect_patch_dim[0]//2
    row_max = center[1] + rect_patch_dim[0]//2
    
    masked_img = data_arr #create a copy if you do not want to change the original file read into it
    if masked_reg_val==1:
        mask = np.zeros(data_arr.shape)
        masked_img_tv = data_arr.copy(); masked_img_tv[row_min:row_max, col_min:col_max, :] = np.nan
        masked_img[row_min:row_max, col_min:col_max, :] = 2*np.pi
        mask[row_min:row_max, col_min:col_max, :] = 2*np.pi
    else:
        mask = np.ones(data_arr.shape)
        masked_img[row_min:row_max, col_min:col_max, :] = 0; masked_img_tv = masked_img
        mask[row_min:row_max, col_min:col_max, :] = 0
    return masked_img, mask, masked_img_tv




def generate_mask_stack(data_arr:np.ndarray, stacked:bool):
    """Creates masked images for a given N dimensional array of images
        Parameters
        ----------
        data_arr : np.ndarray
            is an N dimensional numpy array of the image(s) you wish to mask
        stacked : bool
            has to be True (if you have a stack of identically sized image arrays)
                      False (if your input is a single image array)
        rect_patch_dim : [int,int]
            is a list containing exactly 2 integers for the row(first index) and col(second index) length of the rectagle
        # TODO  center = None
            randomizes center point of the masked region by default. user may enter a fixed center to be used for all the images
        Returns
        -------
        a masked image with the rectangle randomly centered or fixed. The rectangle dimensions are as specified in the inputs
    """

    # Choose a random patch and it corresponding unmask index.
    if stacked:
        for i in range(data_arr.shape[0]):
            data_arr[i] = _generate_mask(data_arr[i]) #we save the mask in data_arr so that we don't have to create another array. Thus, conserving memory
        return data_arr.astype(np.uint8)
    else: return _generate_mask(data_arr)

