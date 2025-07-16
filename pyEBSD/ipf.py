#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 11:03:09 2022

@author: emmanuel
"""

import subprocess
# from shutil import move
import numpy as np
from csv import reader as csv_reader
import platform
import matplotlib.pyplot as plt

from sys import path
path.insert(0, "..") # hack to get module `pyEBSD` in scope
import pyEBSD as ebsd



def saveIPFctf(pipeline: str, 
        output_file: str, path_to_dream3d: bool='./',
        start_num: int=1, im_shape=None):
    '''
    Runs a Dream3d pipeline multiple times automatically, naming the output files sequentially

    ARGS:
    -----
    str: pipeline - Path to the .json file containing the pipeline
    str: output_file - The path to the output file from the pipeline 
    str: path_to_dream3d - Path to the Dream3d folder containing PipelineRunner
    as_numpy - Is true b, output data as .npy array and displays it using matplotlib.pyplot  normalized [0,255])

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
   
    if path_to_dream3d[-1] != '/':
        path_to_dream3d = path_to_dream3d+'/'
        
    
    subprocess.run([path_to_dream3d+'PipelineRunner', '-p', pipeline], startupinfo=startupinfo)
   
    # im = np.genfromtxt(output_file, skip_header=1, delimiter=',', usecols=(0,1,2))
    csv_data = csv_reader( open( output_file, 'rt' ) )
    next(csv_data)
    num_x_cells, num_y_cells, _ = im_shape
    angles = np.empty( (num_x_cells, num_y_cells, 3), dtype=int )
    
    
    for j in range(num_y_cells):
        for i in range(num_x_cells):
            data_row = next(csv_data)
            angles[i,j,0] = float( data_row[0] )
            angles[i,j,1] = float( data_row[1] )
            angles[i,j,2] = float( data_row[2] )
    
    
    
    return np.flipud(angles/255)


def plotIPF(arr: np.ndarray, original_ctf=None):
    ebsd.fileio.save_ang_data_as_ctf('/home/emmanuel/Desktop/EBSD_thesis_codes/pyEBSD/ipfFolder/ipf_gen_temp.ctf',arr, original_file = original_ctf)
    
    im = saveIPFctf(pipeline='/home/emmanuel/Desktop/EBSD_thesis_codes/pyEBSD/ipfFolder/ipf_generator.json', output_file='/home/emmanuel/Desktop/EBSD_thesis_codes/pyEBSD/ipfFolder/ipf_temp_file.csv', path_to_dream3d='/home/emmanuel/Desktop/EBSD_thesis_codes/pyEBSD/ipfFolder/DREAM3D/bin', im_shape=arr.shape)
    plt.imshow(im);plt.axis('off')
    return

def saveIPF(arr: np.ndarray, original_ctf=None):
    ebsd.fileio.save_ang_data_as_ctf('/home/emmanuel/Desktop/EBSD_thesis_codes/pyEBSD/ipfFolder/ipf_gen_temp.ctf',arr, original_file = original_ctf)
    
    im = saveIPFctf(pipeline='/home/emmanuel/Desktop/EBSD_thesis_codes/pyEBSD/ipfFolder/ipf_generator.json', output_file='/home/emmanuel/Desktop/EBSD_thesis_codes/pyEBSD/ipfFolder/ipf_temp_file.csv', path_to_dream3d='/home/emmanuel/Desktop/EBSD_thesis_codes/pyEBSD/ipfFolder/DREAM3D/bin', im_shape=arr.shape)
    return im.astype('float32')

# if __name__=='__main__':
#     plotIPFctf(pipeline='/home/emmanuel/Desktop/EBSD_thesis_codes/pyEBSD/ipfFolder/ipf_generator.json', output_file='/home/emmanuel/Desktop/EBSD_thesis_codes/pyEBSD/ipfFolder/ipf_temp_file.csv', path_to_dream3d='/home/emmanuel/Desktop/EBSD_thesis_codes/pyEBSD/ipfFolder/DREAM3D/bin')
