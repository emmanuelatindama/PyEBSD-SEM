# TODO docstring
from sys import path
path.insert(0, "..") # hack to get module `pyEBSD` in scope
import pyEBSD as ebsd
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy
from numpy import pi
from math import cos, sin, acos
from scipy.spatial.transform import Rotation as Rot
from scipy.ndimage import convolve1d
from scipy.ndimage import median_filter

from skimage import measure
import skimage.morphology as skmorph



def misorientation(a:np.ndarray, b:np.ndarray):
    """
    The inputs of a and b should be vectors (Euler angles-Bunge notation)
    For now, input must be in degrees
    
    step 1: convert a and be to rotation matrices
    step 2: multiply them to obtain a resultant rotation matrix
    step 3: change the rotation matrix to the axis-angle representation
    
    returns:The value of the angle is the deviation 
    """
    
    # Check whether Euler angles are in degrees or radian
    if np.max(a) > 2*np.pi: # if max value is greater than 2pi, then it is in degrees
        a = np.deg2rad(a)   # convert to radians
    if np.max(b) > 2*np.pi: # if max value is greater than 2pi, then it is in degrees
        b = np.deg2rad(b)   # convert to radians
    
    # Orientation 1
    g1_phi1 = np.array([cos(a[0]), sin(a[0]), 0, -sin(a[0]), cos(a[0]), 0, 0, 0, 1]).reshape(3,3)
    g1_Phi  = np.array([1, 0, 0, 0, cos(a[1]), sin(a[1]), 0, -sin(a[1]), cos(a[1])]).reshape(3,3)
    g1_phi2 = np.array([cos(a[2]), sin(a[2]), 0, -sin(a[2]), cos(a[2]), 0, 0, 0, 1]).reshape(3,3)
    g1 = np.matmul(np.matmul(g1_phi2,g1_Phi),g1_phi1)
    
    
    # Orientation 2
    g2_phi1 = np.array([cos(b[0]), sin(b[0]), 0, -sin(b[0]), cos(b[0]), 0, 0, 0, 1]).reshape(3,3)
    g2_Phi  = np.array([1, 0, 0, 0, cos(b[1]), sin(b[1]), 0, -sin(b[1]), cos(b[1])]).reshape(3,3)
    g2_phi2 = np.array([cos(b[2]), sin(b[2]), 0, -sin(b[2]), cos(b[2]), 0, 0, 0, 1]).reshape(3,3)
    g2 = np.matmul(np.matmul(g2_phi2,g2_Phi),g2_phi1) # this is the rotation matrix
    
    
    # Misorientation
    misori = np.matmul(np.transpose(g1),g2)
    
    spur_g = misori[0,0]+misori[1,1]+misori[2,2]; #print(spur_g) # this computes the trace
    w = np.rad2deg(acos((spur_g-1.0000000001)/2))  # this is the angle of misorientation omega
    "we write 1 as 1.0000000001 to avoid floating point errors  that occur when spur_g is >3 due to floating point errors"
    
    return w

def misorientation_map(A:np.ndarray, B:np.ndarray):
    """
    The inputs of a and b should be vectors (Euler angles-Bunge notation)
    For now, input must be in degrees
    
    step 1: convert a and be to rotation matrices
    step 2: multiply them to obtain a resultant rotation matrix
    step 3: change the rotation matrix to the axis-angle representation
    
    The value of the angle is the deviation 
    """
    l,w,h = A.shape
    misori = np.empty((l,w))
    for i in range(l):
        for j in range(w):
            misori[i,j] = misorientation(A[i,j],B[i,j])
    
    return misori


def misorientation_error(A:np.ndarray, B:np.ndarray):
    """
    The inputs of a and b should be vectors (Euler angles-Bunge notation)
    For now, input must be in degrees
    
    step 1: convert a and be to rotation matrices
    step 2: multiply them to obtain a resultant rotation matrix
    step 3: change the rotation matrix to the axis-angle representation
    
    The value of the angle is the deviation 
    """
    l,w,h = A.shape
    misori = np.empty((l,w))
    for i in range(l):
        for j in range(w):
            misori[i,j] = misorientation(A[i,j],B[i,j])
    
    return round(np.deg2rad(np.mean(misori)),3)


def local_misorientation(A:np.ndarray):
    """
    The input, A should be vectors (Euler angles-Bunge notation)
    For now, input must be in degrees
    
    step 1: convert a and be to rotation matrices
    step 2: multiply them to obtain a resultant rotation matrix
    step 3: change the rotation matrix to the axis-angle representation
    
    The value of the angle is the deviation 
    """
    
    l,w,h = A.shape
    horiz_misori = np.zeros_like(A[:,:,0])
    vert_misori  = np.zeros_like(A[:,:,0])
    
    for i in range(l-1):
        for j in range(w-1):
            horiz_misori[i,j] = misorientation(A[i,j],A[i,j+1])
            vert_misori[i,j]  = misorientation(A[i,j],A[i+1,j])
    
    return .5*(horiz_misori + vert_misori)




def SNR(a: np.ndarray, b: np.ndarray):
    """Returns the signal to noise ratio of the two images given.
    Takes 2 images. The first one, b is the altered image and the second one, a is the
    reference image.
    Parameters
    ----------
    a : numpy.ndarray
        An array of shape (M, N, ..., p), where each `q[m, n, ..., :]`
        represents a pixel or voxel.
    Returns
    -------
    float : which represents the signal to noise ratio between the two images
    Raises
    ------
    ValueError
        When the two images have different sizes.
    """
    if a.shape != b.shape:
        raise ValueError("Images have different dimensions")
    a_sq = a**2
    diff = (a-b)**2
    
    div = np.sum(diff)
    num = np.sum(a_sq)
    temp = num/div
    return round(10*np.log10(temp),3)



def mean_l2_error_per_pixel(im1: np.ndarray, im2: np.ndarray):
    if im1.shape != im2.shape:
        raise ValueError("Images have different dimensions");
        
    #m = im1.shape[0]; n = im1.shape[1]; p = im1.shape[2]
    #return round(np.sqrt(np.sum(np.square(im1 - im2))/(m*n*p)), 3)
    return round(np.mean(np.sqrt(np.sum(np.square(im1 - im2), axis=2))), 3)



def eulers_to_quats(e: np.ndarray, order: str) -> np.ndarray:
    # TODO there's a difference between lower-case rotation order ('extrinsic')
    # and upper-case rotation order ('intrinsic'). Note this in the docstring.
    # TODO mention in docstring what the Euler angle range is
    # TODO read scipy.spatial.transform.Rotation documentation and find out more
    # about Euler angle representations (axis order, range depending on axis)
    # TODO validation for input range
    if e.shape[-1] != 3:
        raise ValueError("Last dimension of `e` must be 3 elements long.")

    n_voxels = int( np.prod(e.shape[:-1]) )
    quat_shape = list(e.shape)
    quat_shape[-1] = 4
    
    nan_mask = np.any(np.isnan(e), axis=e.ndim-1)
    e_temp = np.copy(e)
    e_temp[nan_mask, :] = 1 # arbitrary value
    q = Rot.from_euler(order, e_temp.reshape(n_voxels, 3)) \
        .as_quat().reshape(quat_shape)
    q[nan_mask, :] = np.nan
    return q


def quats_to_eulers(q: np.ndarray, order: str) -> np.ndarray:
    # TODO there's a difference between lower-case rotation order ('extrinsic')
    # and upper-case rotation order ('intrinsic'). Note this in the docstring.
    # TODO mention in docstring what the Euler angle range is
    # TODO read scipy.spatial.transform.Rotation documentation and find out more
    # about Euler angle representations (axis order, range depending on axis)
    # TODO validation for inputs being unit quaternions (+- some margin for
    # floating point imprecision in norm calculation)
    if q.shape[-1] != 4:
        raise ValueError("Last dimension of `q` must be 4 elements long.")
    if np.any(np.linalg.norm(q) == 0):
        raise ValueError("`q` must not contain quaternions of norm 0.")
    
    n_voxels = int( np.prod(q.shape[:-1]) )
    euler_shape = list(q.shape)
    euler_shape[-1] = 3
    
    nan_mask = np.any(np.isnan(q), axis=q.ndim-1)
    q_temp = np.copy(q)
    q_temp[nan_mask, :] = 1 # arbitrary nonzero value
    e = Rot.from_quat(q_temp.reshape(n_voxels, 4)) \
        .as_euler(order).reshape(euler_shape)
    e[nan_mask, :] = np.nan
    return e


# TODO should something like [0, 0, 0, 1e-16] be interpreted as [0, 0, 0, 0]
# instead of scaling to [0, 0, 0, 1]?
def unit_quats(q: np.ndarray) -> np.ndarray:
    """Normalize quaternions to unit quaternions.
    
    Takes an array whose last dimension is 4 elements long, interpreted as an
    array of quaternions, and scales each quaternion so that its L2 norm is 1.
    Parameters
    ----------
    q : numpy.ndarray
        An array of shape (M, N, ..., 4), where each `q[m, n, ..., :]`
        represents a quaternion.
    Returns
    -------
    numpy.ndarray
        A copy of `q` with each nonzero quaternion divided by its L2 norm.
        Each quaternion in `q` that contains a NaN part is returned as a fully
        NaN quaternion.
    Raises
    ------
    ValueError
        When the size of the last dimension is not 4.
    """
    norms = np.linalg.norm(q, axis=q.ndim-1, keepdims=True)
    zeros = np.repeat(norms == 0, 4, axis=q.ndim-1)
    if q.shape[-1] != 4:
        raise ValueError("Last dimension of `q` must be 4 elements long.")
    unit_q = q / norms
    unit_q[zeros] = 0
    return unit_q



def clean_360_jumps(arr, window_size = 3):
    """
    The function below takes in an EBSD map of Euler angles and cleans up the jumps that appear in the image 
    due to jumps from 359 degrees to 0 degrees due to periodicity, however appear as sharp differences numerically.
    --------------------------------------------------------------------------------------------------------------
    Input: filename as a string in ctf file format containing Euler angles, and the channel (0,1,or 2) we wish to process.
           File is read as a numpy array.
    
    Output: a numpy array of the preprocessed file is returned. The output range is from 0 to 2*pi, so it has to
            standardized to view properly.
            For experiments sake, we return the input file as well to make it easier for comparing of results
    --------------------------------------------------------------------------------------------------------------
    Example:
    test = preprocess("Synthetic_test_noisy.ctf")
    plt.imshow(test/(2*pi))
    
    """
    num_of_bins = 300 
        
    separated_mean_kernel = np.ones(window_size) / window_size
    for chan in [0,2]:
        e = arr[:,:,chan]
        # get E[X^2]
        local_mean_of_square = convolve1d(e**2, separated_mean_kernel, axis=0)
        
        # get (E[X])^2
        local_mean = convolve1d(e, separated_mean_kernel, axis=0)
        square_of_local_mean = local_mean**2
        
        channelwise_variance_per_pixel = local_mean_of_square - square_of_local_mean
        # std_dev(X) = sqrt(E[X^2] - (E[X])^2) compute standard deviation
        for i in range(e.shape[0]):
            for j in range(e.shape[1]):
                if channelwise_variance_per_pixel[i,j] < 0:
                    channelwise_variance_per_pixel[i,j] = -1*channelwise_variance_per_pixel[i,j]
        
        try:
            # std_dev(X) = sqrt(E[X^2] - (E[X])^2) compute standard deviation
            grayscale_std_dev = np.sqrt(channelwise_variance_per_pixel)
            hist,bins,patches = plt.hist(grayscale_std_dev.flatten(), num_of_bins); plt.clf();
            
            argmax_hist = bins[np.argmax(hist)] #this value is used in the next block of code as the cut-off threshold
        
            def temp_gaussian_tail(data, peak = argmax_hist):
        
                hist,patches = np.histogram(data.flatten(), num_of_bins)    
                #smoothing the histogram
                hist_smoothing = [(hist[i-2]+hist[i-1]+hist[i]+hist[i+1]+hist[i+2])/5 for i in range(2, num_of_bins-2)]
                
                i=np.where(patches==[peak])[0][0]
                while hist_smoothing[i]>hist_smoothing[i+1]:
                    i+=1
                tail= patches[i+1]
                num_bins = int(np.sqrt(sum(hist[: list(patches).index(tail)])))
                return num_bins, tail
        
            def largest_patch(mask):
                """
                Computes the length (number of pixels) of the largest patch in the connected masked
                regions of the mask provided. The masked regions have boolean value 'True', while 
                the every other part is 'False'. If the largest connected region has say, 20 pixels
                it returns 20
                ----------------------------------------------------------------------------------
                input: A mask of boolean values
                returns: the number of maximum number of pixels in the largest connected region.
                """
                max_value = 0
                labels, num_of_components = measure.label(mask, return_num=True)
                for prop in measure.regionprops(labels):
                    coordslist_of_region = prop.coords
                    max_value = max(sum([1 for index in coordslist_of_region]), max_value)
                return max_value
        
            def count_num_in_arr(X, num, direction):
                """
                Counts the number of elements in the 'direction' of 'num'.
                ----------------------------------------------------------
                Example:
                X = np.array([[1,0,3],
                              [2,5,7],
                              [-2,4,1]])
                count_num_in_arr(X, 5, 'greater than')
                returns 2 since 5 and 7 are greater than or equal to 5
                """
                count=0
                if direction=='greater than':
                    for i in range(X.shape[0]):
                        for j in range(X.shape[1]):
                            if X[i,j]>=num:
                                count+=1
                elif direction=='less than':
                    for i in range(X.shape[0]):
                        for j in range(X.shape[1]):
                            if X[i,j]<=num:
                                count+=1
                return count
              
            num_bins, temp_threshold1 = temp_gaussian_tail(grayscale_std_dev)
            
            def gaussian_tail(data):
        
                hist,patches = np.histogram(data.flatten(), num_bins)
                
                #smoothing the histogram
                hist_smoothing = [(hist[i-2]+hist[i-1]+hist[i]+hist[i+1]+hist[i+2])/5 for i in range(2, num_bins-2)]

                
                i=np.argmax(hist)
                
                while hist_smoothing[i]>hist_smoothing[i+1]:
                    i+=1
                
                return patches[i+1]
                    
            temp_threshold2 = gaussian_tail(grayscale_std_dev)
             
            threshold = min(temp_threshold1, temp_threshold2)
        except:
            threshold = 20*pi/180
        
        mask_0 = ((e <= threshold) & (grayscale_std_dev > threshold))
        mask_360 = ((e >= 2*pi-threshold) & (grayscale_std_dev > threshold))
#         mask_std_dev = grayscale_std_dev > threshold
        
        mask1 = ((e <= threshold) | (e >= 2*pi-threshold))
        mask2 = skmorph.remove_small_holes(mask1)
        mask3 = skmorph.binary_erosion(mask2)
        mask4 = skmorph.remove_small_objects(mask3, 3)
        final_mask = skmorph.binary_dilation(mask4)
    
        patch_length = largest_patch(final_mask)
    
        m=min(int((math.ceil(np.sqrt(2*int(patch_length/2))+1)-1)/2),5)
        
        combined_mask = False*np.ones_like(mask1)
        
        # this handles the index errors that will occur at the boundaries
        for i in range(e.shape[0]):
            if i-m<0:
                il=m
            else:
                il=i
            for j in range(e.shape[0]):
                if j-m<0:
                    jl=m
                else:
                    jl=j
                if mask_0[i,j]==True and (True in mask_360[il-m:i+m, jl-m:j+m]):
                    combined_mask[i,j]=True
                if mask_360[i,j]==True and (True in mask_0[il-m:i+m, jl-m:j+m]):
                    combined_mask[i,j]=True
        e_temp = e.copy()
         
        threshold = max(threshold, 20*pi/180)
        # Cleaning
        for i in range(e.shape[0]):
            if i-m<0:
                il=m
            else:
                il=i
            for j in range(e.shape[1]):
                if j-m<0:
                    jl=m
                else:
                    jl=j
                if combined_mask[i,j]==True:
                    num_vals_close_to_360 = count_num_in_arr(e[il-m:i+m, jl-m:j+m],2*pi-threshold, 'greater than')
                    num_vals_close_to_0 = count_num_in_arr(e[il-m:i+m, jl-m:j+m], threshold, 'less than')
                    if (num_vals_close_to_0 >= num_vals_close_to_360) and (e[i,j] > (2*pi-threshold)):
                        e_temp[i,j] = 2*pi - e[i,j]
                    elif (num_vals_close_to_360 >= num_vals_close_to_0) and (e[i,j] < threshold):
                        e_temp[i,j] = 2*pi - e[i,j]
        arr[:,:,chan] = e_temp
        
    return arr



def clean_discontinuities(ebsd_data):
    try:
        cleaned = clean_360_jumps(ebsd_data)
    except ValueError:
        print('Oops! noise is too low for preprocessing')
        cleaned = ebsd_data
    return cleaned



def fill_isolated_with_median(u, num_iterations=3):
    """
        Identifies the isolated points in and image and inpaints them.
        --------------------------------------------------------------
        Input:
        u : is the input image 
        num_iterations: the number of times we will repeat the inpainting
                        process on the image. The default value is 20
        
        Output: 
        returns the inpainted data.
        --------------------------------------------------------------
    """
    """inpainting isolated points"""
    f1 = np.asarray([[1, 1, 1],
                     [1, 0, 1],
                     [1, 1, 1]])
    
    f2 = np.asarray([[1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 1],
                     [1, 1, 0, 1, 1],
                     [1, 1, 0, 1, 1],
                     [1, 1, 1, 1, 1]])
    
    f3 = np.asarray([[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 0, 0, 1],
                     [1, 1, 0, 1, 1],
                     [1, 1, 1, 1, 1]])
    
    f4 = np.asarray([[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 0, 0, 1],
                     [1, 1, 1, 0, 1],
                     [1, 1, 1, 1, 1]])
    
    f5 = np.asarray([[1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 1],
                     [1, 1, 0, 0, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]])

    f6 = np.asarray([[1, 1, 1, 1, 1],
                     [1, 1, 0, 1, 1],
                     [1, 1, 0, 0, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]])
    
    u_inpainted = u.copy()
    inpainted = np.zeros_like(u)
    for k in range(num_iterations):
        for i in range(u_inpainted.shape[2]):
            A = u_inpainted[:,:,i]
            """condition 1: value greater or less than all of its 3x3 neighbors by magnitude of 0.05 or more"""
            condition1a = A - scipy.ndimage.maximum_filter(A, footprint=f1, mode='constant') > 0.05
            condition1b = A - scipy.ndimage.minimum_filter(A, footprint=f1, mode='constant') < -0.05
            """condition 2: value greater or less than all of its 5x5 neighbors by magnitude of 0.05 or more"""
            condition2a = A - scipy.ndimage.maximum_filter(A, footprint=f2, mode='constant') > 0.05
            condition2b = A - scipy.ndimage.minimum_filter(A, footprint=f2, mode='constant') < -0.05            
            "The value is 1 if condition is true. Else, the value is 0."
            result = np.zeros(A.shape)
            result[condition1a|condition1b|condition2a|condition2b]=1
       
            inpainted[:,:,i] = median_filter(A,3)
            for x in range(inpainted.shape[0]):
                for y in range(inpainted.shape[1]):
                    if result[x,y]==1:
                        u_inpainted[x,y,i] = inpainted[x,y,i]
    for k in range(num_iterations):
        for i in range(u_inpainted.shape[2]):
            A = u_inpainted[:,:,i]
            """condition 1: value greater or less than all of its 3x3 neighbors by magnitude of 0.05 or more"""
            condition1a = A - scipy.ndimage.maximum_filter(A, footprint=f1, mode='constant') > 0.05
            condition1b = A - scipy.ndimage.minimum_filter(A, footprint=f1, mode='constant') < -0.05
            """condition 2: value greater or less than all of its 5x5 neighbors by magnitude of 0.05 or more"""
            condition2a = A - scipy.ndimage.maximum_filter(A, footprint=f2, mode='constant') > 0.05
            condition2b = A - scipy.ndimage.minimum_filter(A, footprint=f2, mode='constant') < -0.05
            """condition 3: value greater or less than all of its 5x5 neighbors by magnitude of 0.05 or more"""
            condition3a = A - scipy.ndimage.maximum_filter(A, footprint=f3, mode='constant') > 0.05
            condition3b = A - scipy.ndimage.minimum_filter(A, footprint=f3, mode='constant') < -0.05
            """condition 4: value greater or less than all of its 5x5 neighbors by magnitude of 0.05 or more"""
            condition4a = A - scipy.ndimage.maximum_filter(A, footprint=f4, mode='constant') > 0.05
            condition4b = A - scipy.ndimage.minimum_filter(A, footprint=f4, mode='constant') < -0.05
            """condition 5: value greater or less than all of its 5x5 neighbors by magnitude of 0.05 or more"""
            condition5a = A - scipy.ndimage.maximum_filter(A, footprint=f5, mode='constant') > 0.05
            condition5b = A - scipy.ndimage.minimum_filter(A, footprint=f5, mode='constant') < -0.05
            """condition 6: value greater or less than all of its 5x5 neighbors by magnitude of 0.05 or more"""
            condition6a = A - scipy.ndimage.maximum_filter(A, footprint=f6, mode='constant') > 0.05
            condition6b = A - scipy.ndimage.minimum_filter(A, footprint=f6, mode='constant') < -0.05
                      
            "The value is 1 if condition is true. Else, the value is 0."
            result = np.zeros(A.shape)
            result[condition1a|condition1b|condition2a|condition2b|condition3a|condition3b]=np.nan
            result[condition4a|condition4b|condition5a|condition5b|condition6a|condition6b]=np.nan
            
            inpainted[:,:,i] = median_filter(A,3)
            for x in range(inpainted.shape[0]):
                for y in range(inpainted.shape[1]):
                    if result[x,y]==1:
                        u_inpainted[x,y,i] = inpainted[x,y,i]       
    return u_inpainted



def apply_median_filter(arr: np.ndarray):
    med_filter = np.zeros_like(arr)
    for i in range(arr.shape[2]):
        med_filter[:,:,i] = median_filter(arr[:,:,i],3)
    return med_filter



"""Inpainting isolated points with ebsd flow"""
def inpaint_isolated_pts(u, num_iterations=3, threshold=.05):
    """
        Identifies the isolated points in and image and inpaints them.
        --------------------------------------------------------------
        Input:
        u : is the input image
        
        num_iterations: the number of times we will repeat the inpainting
                        process on the image. The default value is 20
        Output: 
        returns the inpainted data.
        --------------------------------------------------------------
    """
    """inpainting isolated points"""
    f1 = np.asarray([[1, 1, 1],
                     [1, 0, 1],
                     [1, 1, 1]])

    f2 = np.asarray([[1, 1, 1, 1, 1],
                     [1, 0, 0, 0, 1],
                     [1, 1, 0, 1, 1],
                     [1, 1, 0, 1, 1],
                     [1, 1, 1, 1, 1]])
    
    f3 = np.asarray([[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 0, 0, 1],
                     [1, 1, 0, 1, 1],
                     [1, 1, 1, 1, 1]])
    
    f4 = np.asarray([[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 0, 0, 1],
                     [1, 1, 1, 0, 1],
                     [1, 1, 1, 1, 1]])
    
    f5 = np.asarray([[1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 1],
                     [1, 1, 0, 0, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]])

    f6 = np.asarray([[1, 1, 1, 1, 1],
                     [1, 1, 0, 1, 1],
                     [1, 1, 0, 0, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]])
    
    u_missing = u.copy()
    for k in range(num_iterations):
        for i in range(u_missing.shape[2]):
            A = u_missing[:,:,i]
            """condition 1: value greater or less than all of its 3x3 neighbors by magnitude of 0.05 or more"""
            condition1a = A - scipy.ndimage.maximum_filter(A, footprint=f1, mode='constant') > threshold
            condition1b = A - scipy.ndimage.minimum_filter(A, footprint=f1, mode='constant') < -threshold

            "The value is 1 if condition is true. Else, the value is 0."
            result = np.zeros(A.shape)
            result[condition1a|condition1b]=np.nan
            #Set isolated values to nan
            u_missing[:,:,i] =  u_missing[:,:,i] + result
        temp = ebsd.inpaint(u_missing, delta_tolerance=1e-5, on_quats=False)
        u_missing = temp
        
    for k in range(num_iterations):
        for i in range(u_missing.shape[2]):
            A = u_missing[:,:,i]
            """condition 1: value greater or less than all of its 3x3 neighbors by magnitude of 0.05 or more"""
            condition1a = A - scipy.ndimage.maximum_filter(A, footprint=f1, mode='constant') > threshold
            condition1b = A - scipy.ndimage.minimum_filter(A, footprint=f1, mode='constant') < -threshold
            """condition 2: value greater or less than all of its 5x5 neighbors by magnitude of 0.05 or more"""
            condition2a = A - scipy.ndimage.maximum_filter(A, footprint=f2, mode='constant') > threshold
            condition2b = A - scipy.ndimage.minimum_filter(A, footprint=f2, mode='constant') < -threshold
            """condition 3: value greater or less than all of its 5x5 neighbors by magnitude of 0.05 or more"""
            condition3a = A - scipy.ndimage.maximum_filter(A, footprint=f3, mode='constant') > threshold
            condition3b = A - scipy.ndimage.minimum_filter(A, footprint=f3, mode='constant') < -threshold
            """condition 4: value greater or less than all of its 5x5 neighbors by magnitude of 0.05 or more"""
            condition4a = A - scipy.ndimage.maximum_filter(A, footprint=f4, mode='constant') > threshold
            condition4b = A - scipy.ndimage.minimum_filter(A, footprint=f4, mode='constant') < -threshold
            """condition 5: value greater or less than all of its 5x5 neighbors by magnitude of 0.05 or more"""
            condition5a = A - scipy.ndimage.maximum_filter(A, footprint=f5, mode='constant') > threshold
            condition5b = A - scipy.ndimage.minimum_filter(A, footprint=f5, mode='constant') < -threshold
            """condition 6: value greater or less than all of its 5x5 neighbors by magnitude of 0.05 or more"""
            condition6a = A - scipy.ndimage.maximum_filter(A, footprint=f6, mode='constant') > threshold
            condition6b = A - scipy.ndimage.minimum_filter(A, footprint=f6, mode='constant') < -threshold
                      
            "The value is 1 if condition is true. Else, the value is 0."
            result = np.zeros(A.shape)
            result[condition1a|condition1b|condition2a|condition2b]=np.nan
            result[condition3a|condition3b|condition4a|condition4b|condition5a|condition5b|condition6a|condition6b]=np.nan

            #Set isolated values to nan
            u_missing[:,:,i] =  u_missing[:,:,i] + result
        temp = ebsd.inpaint(u_missing, delta_tolerance=1e-5, on_quats=False)
        u_missing = temp
        
    return u_missing