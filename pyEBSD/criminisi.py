
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:36:53 2023

"""
import numpy as np
import scipy.ndimage as img
from numba import njit

from time import time #DEBUG
from skimage.color import rgb2gray #TODO: take by hand
import cv2 as cv
import imageio #For the gif
from PIL import Image


def get_boundary(im: np.ndarray) -> np.ndarray:
    """ get_boundary
    Inputs:
        im: An np.array. The mask of image 
    Returns:
        A boolean np.array that is the shape of the inputted image
        This array is all false except on the boundary points, which are true
    """
    return img.binary_erosion(im, np.full((3, 3), True)) ^ im


def patch_slice(point: tuple, half_patch_size: int) -> tuple:
    """patch_slice
    Inputs:
        point: a tuple. the index of the center point that we want the patch of
        half_patch_size: and int. the size of the desired patch divided by 2. 
    Returns:
        A tuple that contains all four corner indexes of the patch.
    """
    return (slice(point[0] - half_patch_size, point[0] + half_patch_size+1),
            slice(point[1] - half_patch_size, point[1] + half_patch_size+1))


def patch(im: np.ndarray, point: tuple, half_patch_size: int):
    """patch
    Inputs:
        im: np.ndarray, this is the array that we will pull the patch from. 
        i.e. it could pull that patch from the image we are filling in or pull
        the patch from the working mask so we know what is valid.
        point: A tuple. This is the center point of the patch that we want. 
        half_patch_size: an int, this is the size of the patch that we want. 
    Returns:
        The patch centered at the point in the image
    """
    return im[patch_slice(point, half_patch_size)]


def get_confidence(prev_confidences: np.ndarray, point: tuple,
                   half_patch_size: int, patch_area: int) -> float:
    """
    Inputs: 
        prev_confidences: An np.array, which is the array of the confidences for
        all points in the image. Note: expects `prev_confidences` to be 0 in
        the unfilled region.
        point: a tuple, this is the point that we want to get the confidence of. 
        half_patch_size: an int, this is the size of the patch we are using /2
        patch_area: an int, this is the area of the patch we are using
    Returns:
        The sum of all the confidences in the patch, which will then become 
        the confidence of the inputted point,
    """
    return np.sum(patch(prev_confidences, point, half_patch_size)) / patch_area


def get_confidences(prev_confidences: np.ndarray, boundary: list,
                    half_patch_size: int, patch_area: int
                   ) -> list:
    """get_confidences
    Inputs:
        prev_confidences: np.array the confidence matrix from the previous iteration    
        boundary: a list of all points at the boundary
        half_patch_size: an int, this is the size of the patch we are using /2
        patch_area: an int, this is the area of the patch we are using
    Does:
        Sets up inputs for the get_confidence method. 
    Returns:
        A list of the confidences at all of the boundary points.
    """
    arg1 = [prev_confidences]*len(boundary)
    arg2 = boundary
    arg3 = [half_patch_size]*len(boundary)
    arg4 = [patch_area]*len(boundary)
    return list(map(get_confidence, arg1, arg2, arg3, arg4))



def calc_normal_matrix(in_mask: np.ndarray) -> np.ndarray:
    """calc_normal_matrix
    Inputs:
        in_mask: the working mask of the image
    Does: 
    	Uses a sobel matrix to find the boundary of the mask
        Calculates the strngth of isophotes throughout the mask
    Returns:
        A matrix that contains the normal at each point in the mask
    """
    x_kernel = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
    y_kernel = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])
    
    x_normal = img.filters.convolve(in_mask.astype(float), x_kernel)
    y_normal = img.filters.convolve(in_mask.astype(float), y_kernel)
    normal = np.dstack((x_normal, y_normal))
    
    height, width = normal.shape[:2]
    
    norm = np.sqrt(y_normal**2 + x_normal**2) \
                 .reshape(height, width, 1) \
                 .repeat(2, axis=2)
    norm[norm == 0] = 1
    unit_normal = normal/norm
    return unit_normal
 

def calc_gradient_matrix(im: np.ndarray, unfilled_mask: np.ndarray, boundary: tuple,
             half_patch_size: int, patch_size: int) -> np.ndarray:
    """
    Inputs:
        im: np.ndarray, the current working image
        unfilled_mask: np.ndarray, the current working mask
        boundary: tuple, list of boundary points as a tuple, 
        half_patch_size: int, the desired size of the patch /2
        patch_size: int, the size of the patch
    Does:
        Calculates all of the gradients for each point in the image, sums 
        gradients in all three channels on x and y planes
    Returns:
        max_gradient: np.array, that contains the maximum gradient of each
        patch centered at the indicated point.
    """              
    height, width = im.shape[:2]
    boundary_list = list(zip(boundary[0],boundary[1]))
    
    
    """
    #MGW attemp at changing gradient. 
    #This gives worse results than the current method.
    #The current method however is wrong in the context of EBSD data. 
    temp_image = np.copy(im)
    temp_image[unfilled_mask == 1] = None
    gradient = np.nan_to_num(np.array(np.gradient(temp_image)))
    if gradient.ndim == 4:
        gradient = gradient[:2,...]
        gradient = np.nan_to_num(np.sum(np.absolute(gradient), axis = 3))
    """
    
    grey_image = rgb2gray(im) #TODO: take gradient by hand
    grey_image[unfilled_mask == 1] = None
    gradient = np.nan_to_num(np.array(np.gradient(grey_image)))
    
    
    
    gradient_val = np.sqrt(gradient[0]**2 + gradient[1]**2)
    max_gradient = np.zeros([height, width, 2])
    
    for point in boundary_list:
        patch_y_gradient = patch(gradient[0], point, half_patch_size)
        patch_x_gradient = patch(gradient[1], point, half_patch_size)
        patch_gradient_val = patch(gradient_val, point, half_patch_size)
        
        patch_max_pos = np.unravel_index(
            patch_gradient_val.argmax(), patch_gradient_val.shape)
        
        max_gradient[point[0], point[1], 0] = \
                patch_y_gradient[patch_max_pos]
        max_gradient[point[0], point[1], 1] = \
            patch_x_gradient[patch_max_pos] 
    
    return max_gradient
    

def get_data(im: np.ndarray, unfilled_mask: np.ndarray, boundary: tuple,
             half_patch_size: int, patch_size: int) -> list:
    """
    Inputs:
    	im: np.ndarray, the current working image
    	unfilled_mask: np.ndarray, the current working mask
    	boundary: tuple, list of boundary points as a tuple, 
    	half_patch_size: int, the desired size of the patch /2
    	patch_size: int, the size of the patch
    Does:
    	Calls functions to calculate the norm of the mask and gradient of the image
        returns the data at these points
    Returns:
        data: a list of all of the data values at the points on the boundaries
    """ 
    normal = calc_normal_matrix(unfilled_mask)
    gradient = calc_gradient_matrix(
        im, unfilled_mask, boundary, half_patch_size, patch_size)
    normal_gradient = normal*gradient 
    data_mtrx = np.sqrt(
        normal_gradient[:, :, 0]**2 + normal_gradient[:, :, 1]**2) + 0.001
    boundary_list = list(zip(boundary[0],boundary[1]))
    data = []
    for point in boundary_list:
        data.append(data_mtrx[point])
    return data
    
  
def get_priority_point(boundary: list, confidences: list, data: list):
    """get_priority_point
    Inputs:
    	boundary: list, the list of points along the boundary of the damaged region
    	confidences: list, the confidences of the points along damaged region 
    	data: list, the datat of the points along the damaged region

    Returns:
    	The point with the highest priority,
    Note: Priority for a point is confidence * data at that point
    """ 
    priorities = np.array(confidences) * np.array(data)
    target_ind = np.argmax(priorities)
    return (boundary[target_ind], confidences[target_ind])


"""Patch Distance Functions
Inputs:
    source_patch_flat: np.ndarray, the patch we will compare the target to
    target_patch_flat: np.ndarray, the patch we are looking to fill in
    filled_patch_flat: np.ndarray, the indices of the valid pixels in the target patch
Returns:
    The distance between the target and source patches, as defined by the particular distance function
Note: 
    This has been optimized for speed using @njit.
    We always take the patch with the smallest distance to be the best.  So some of these
    distance functions are modified slightly from their original definition to ensure that
    maximum similarity occurs at the minimum values, rather than maximum (noted in function
    specific comments)
"""

#Sum Squared Error
@njit
def SSE(source_patch_flat: np.ndarray, target_patch_flat: np.ndarray,
         filled_patch_flat: np.ndarray) -> float:
        return np.sum(
            (source_patch_flat - target_patch_flat)[filled_patch_flat]**2
        )

#Mean Squared Error
@njit
def MSE(source_patch_flat: np.ndarray, target_patch_flat: np.ndarray,
        filled_patch_flat: np.ndarray) -> float:
    return np.mean(
        (source_patch_flat - target_patch_flat)[filled_patch_flat]**2
        )


#Cosine similarity is bounded on [-1,1], with maximum similarity occuring at 1
#So we multiply by -1 to make maximum similarity occur at the minimum value
@njit
def cosine_similarity(source_patch_flat: np.ndarray, target_patch_flat: np.ndarray,
        filled_patch_flat: np.ndarray) -> float:
    return -np.sum(source_patch_flat[filled_patch_flat] * target_patch_flat[filled_patch_flat])/(\
           np.sqrt(np.sum(source_patch_flat[filled_patch_flat]**2)) *\
           np.sqrt(np.sum(target_patch_flat[filled_patch_flat]**2)) + 1e-8)



#Structural Similarity Index Measure
#L is the 'dynamic range' of the pixels (max pixel value - min pixel value)
#For a normal image this would be 255, but for EBSD data it should be 359
#c1 and c2 ensure the denominator is never 0
#More similar images have higher SSIM, so multiply whole thing by -1
@njit
def SSIM(source_patch_flat: np.ndarray, target_patch_flat: np.ndarray,
        filled_patch_flat: np.ndarray) -> float:
    L = 359
    k1 = .01
    k2 = .03
    c1 = (k1*L)**2
    c2 = (k2*L)**2
    mux = np.mean(source_patch_flat[filled_patch_flat])
    muy = np.mean(target_patch_flat[filled_patch_flat])

    #Can't just pass [source, target] to np.cov when using @njit
    #So explicitly construct it as an ndarray
    cov_mat = np.empty((2, np.count_nonzero(filled_patch_flat)))
    cov_mat[0] = source_patch_flat[filled_patch_flat]
    cov_mat[1] = target_patch_flat[filled_patch_flat]
    
    cov = np.cov(cov_mat)
    covx = cov[0,0]
    covy = cov[1,1]
    covxy = cov[1,0]
    return -( (2*mux*muy + c1)*(2*covxy+c2) )/( (mux**2+muy**2+c1)*(covx+covy+c2) )


#KL divergence can be +/-, with the best match around 0, so this is actually absolute value of KL divergence
@njit
def KLdivergence(source_patch_flat: np.ndarray, target_patch_flat: np.ndarray,
                 filled_patch_flat: np.ndarray) -> float:
    return np.abs( np.sum( target_patch_flat[filled_patch_flat] * np.log(\
        target_patch_flat[filled_patch_flat] / source_patch_flat[filled_patch_flat]) ) )


@njit
def logcosh(source_patch_flat: np.ndarray, target_patch_flat: np.ndarray,
            filled_patch_flat: np.ndarray) -> float:
    return np.sum( np.log( np.cosh( source_patch_flat[filled_patch_flat] - target_patch_flat[filled_patch_flat] )))


"""dist
Inputs:
    source_patch_flat: np.ndarray, the patch we will compare the target to
    target_patch_flat: np.ndarray, the patch we are looking to fill in
    filled_patch_flat: np.ndarray, the indices of the valid pixels in the target patch
Returns:
    The Sum of Squared Distances between all of the known pixels in the target and source
        divided by the number of known pixels
Note: 
    This has been optimized for speed using @njit
    This is our new calculation 
"""
@njit
def our_dist(source_patch_flat: np.ndarray, target_patch_flat: np.ndarray,
         filled_patch_flat: np.ndarray) -> float:
        return np.sum(
            (source_patch_flat - target_patch_flat)[filled_patch_flat]**2
        ) / np.count_nonzero(filled_patch_flat)

    
"""get_best_exmeplar
Inputs:
    target_point: tuple, the point we are looking to inpaint
    im: np.ndarray, the working image
    original_mask: np.ndarray, the mask that we started with initially
    working_mask: np.ndarray, the mask that indicates what is currently filled and what is not
    patch_size: int, the patch size used for filling
    compare_psz: int = 0, if the user wishes to use a larger patch size for comparison that is used here,
        defaults to 0, as this is not in the original Criminisi paper
    our_distance: bool = True, if we are using our formula for distance this is true, 
        made togglable so it can be turned off as described in original Criminisi paper
    restrict_search: bool = False if we are restricting the search region this is true, 
        made togglable so it can be "turned off" 
    
Does:
    Initializes Search region, calculates distance of every patch
Returns:    
    The patch from the image that has the lowest distance
"""
def get_best_exemplar(target_point: tuple, im: np.ndarray,
                      original_mask: np.ndarray, working_mask: np.ndarray,
                      patch_size: int, compare_psz: int = 0,
                      distance_metric = SSE, restrict_search: bool = False,
                      euclidean_penalty: bool = True) -> np.ndarray:
    
    half_patch_size = compare_psz //2
    
    temp_mask = np.ones_like(im, dtype=bool)
    
    filled_patch_flat = \
        ~patch(temp_mask, target_point, half_patch_size).ravel()
    
    source_point_mask = ~original_mask
    source_point_mask = img.binary_erosion(
        source_point_mask, np.full((3, 3), True), half_patch_size
    )
    if restrict_search:
        search_restriction = 5
        
        row_start = max(target_point[0] - im.shape[0]//search_restriction + half_patch_size, 0)
        row_end = min (target_point[0] + im.shape[0]//search_restriction - half_patch_size, 
            im.shape[0] - half_patch_size )
        
        col_start = max(target_point[1] - im.shape[1]//search_restriction + half_patch_size, 0)
        col_end = min (target_point[1] + im.shape[1]//search_restriction - half_patch_size, 
            im.shape[1] - half_patch_size )
        
        if row_start>= row_end or col_start>= col_end: 
            raise ValueError("Search Restricted too much!") 
        
        source_point_mask[:row_start, :] = False
        source_point_mask[row_end:, :]  = False
        source_point_mask[:, :col_start] = False
        source_point_mask[:, col_end:]  = False
        
        search_region = (slice(row_start,row_end), slice(col_start,col_end))
    
    else:
        source_point_mask[:half_patch_size, :] = False
        source_point_mask[-half_patch_size:, :]  = False
        source_point_mask[:, :half_patch_size] = False
        source_point_mask[:, -half_patch_size:]  = False
        
        search_region = (slice(half_patch_size,-half_patch_size), slice(half_patch_size,-half_patch_size))
        
    source_points = source_point_mask.nonzero()
    source_points = list(zip(source_points[0], source_points[1]))
    
    if im.ndim == 3:
        distances = np.empty(im.shape)
        for channel in range(im.shape[2]):
            target_patch_flat = \
                patch(im[..., channel], target_point, half_patch_size).ravel()

            distances[search_region[0],search_region[1], channel] = img.generic_filter(
                im[search_region[0],search_region[1], channel], distance_metric, size=(compare_psz, compare_psz),
                extra_arguments=(target_patch_flat, filled_patch_flat)
            )
        distances = np.sum(distances, axis=2)
    elif im.ndim == 2:
        target_patch_flat = patch(im, target_point, half_patch_size).ravel()
        distances = img.generic_filter(
            im, distance_metric, size=(compare_psz, compare_psz),
            extra_arguments=(target_patch_flat, filled_patch_flat)
        )

    #TODO: Different metrics output on different scales, so the
    #Euclidean distance is probably having different levels of effect
    #for different metrics.
        #But since its weighted equally within the same metric, its
        #probably fine?
    if euclidean_penalty:
        for row in range(im.shape[0]):
            for col in range(im.shape[1]):
                if not source_point_mask[row, col]:
                    continue
                distances[row,col] +=  np.sqrt(
                    (target_point[0] - row)**2 + (target_point[1] - col)**2
                )/ np.sqrt(im.shape[0]**2 + im.shape[1]**2)
    
    source_point = source_points[np.argmin(distances[source_point_mask])]
    print("Source: " + str(source_point))
    print("Source Metric:" + str(np.min(distances[source_point_mask])))
    return patch(im, source_point, patch_size//2)
    

"""fill_patch
Inputs:
    im: np.ndarray, the working image
    target_point: tuple, the point we are filling in
    source_patch: np.ndarray, the patch we are using for filling
    unfilled_mask: np.ndarray, the working mask indicating what is and isnt filled in
    confidences: np.ndarray, array indicating the confidences of all pixels in the image
    target_confidence: float,the confidence of the target pixel
    half_patch_size: int, the size of the patch used for filling/2
Does:
    Updates the unfilled image, working mask and confidences
Note:
    modifies some inputs in-place
"""
def fill_patch(im: np.ndarray, target_point: tuple, source_patch: np.ndarray,
               unfilled_mask: np.ndarray, confidences: np.ndarray,
               target_confidence: float, half_patch_size: int,
               onion: bool = False):
    
    target_patch_slice = patch_slice(target_point, half_patch_size)
    unfilled_patch = unfilled_mask[target_patch_slice] 
    
    im[target_patch_slice][unfilled_patch] = source_patch[unfilled_patch]
    unfilled_mask[target_patch_slice][unfilled_patch] = False
    
    if not onion:
        confidences[target_patch_slice] = target_confidence


"""inpaint
(This is essentially the main function)
Inputs:
    im: np.ndarray, the original image we are looking to have repaired
    original_mask: np.ndarray, indiciates damaged and known regions
    patch_size: int = 3, the size of patch used throughout the process
    compare_increase: int =0, if we would like to use a larger patch during filling, the increase in 
        size is indicated here
    distance_metric: (np.ndarray, np.ndarray, np.ndarray)->float, the distance metric to use when comparing patches
    restrict_search: bool =False, if True use a restricted search region centered at the target patch
    euclidean_penalty: bool=True, if we would like to add the Euclidean distance from a canidate patch to the fill region to the distance metric
    onion: bool=False, if we wish to use onion layering rather than the Criminisi priority for fill order
    show_movie: bool = False, if True then show the live image during filling
    save_movie: bool = False, if True then save fill progression as gif 
Throws:
    ValueError: If the patch size is not odd
    ValueError: if the comparison size is not even
    ValueError: If the color values of the pixels are not between 0 and 1
Does:
    Facilitates the inpainting 
Returns:
    The final inpainted image

"""

# `original_mask` should be a boolean mask
def inpaint(im: np.ndarray, original_mask: np.ndarray, patch_size: int = 3, 
        compare_increase: int =0, restrict_search: bool =False,
        euclidean_penalty: bool = True, distance_metric = SSE,
        show_movie: bool = False, onion: bool = False,
        save_movie: bool = False) -> np.ndarray:
    
    if patch_size % 2 != 1: raise ValueError("`patch_size` must be odd.")
    if compare_increase % 2 != 0: raise ValueError("`compare_increase` must be even.")
    
    if not np.all((0 <= im) & (im <= 1)): #Necessary
        raise ValueError("All values in `im` must be on the interval [0, 1].")
    
    im = im.copy()
    
    half_patch_size = patch_size // 2
    patch_area = patch_size**2
    #Unfilled is the "working mask", indicating what is left to be filled in
    unfilled = original_mask.copy()
    
    #Confidences is an array indicating the confidence of each point
    confidences = (~original_mask).copy().astype(float)
    
    compare_psz = patch_size + compare_increase
    
    #Used for Gif of filling
    if save_movie:
        movie = []
    

    total_exemplar_time = 0 # DEBUG
    total_data_time = 0 # DEBUG
    total_confidence_time = 0 # DEBUG
    t_start = time() # DEBUG
    iter_counter = 0 # DEBUG

    #Main loop, continue filling until nothing left to fill
    while np.any(unfilled): 
        iter_counter += 1
        print(f"\nStarting iteration: {iter_counter}")
        
        #Show progress during filling
        if show_movie:
            cv.imshow('live progress', im)
            cv.waitKey(5)
        if save_movie:
            #movie.append(np.copy(im))
            movie.append(Image.fromarray((np.copy(im)*255).astype(np.uint8)))
        
        boundary_tuple = get_boundary(unfilled).nonzero()
        boundary_list = list(zip(boundary_tuple[0], boundary_tuple[1]))

        #Follow Criminisi fill order
        if not onion:
            t_confidence_start = time() # DEBUG
            boundary_confidences = get_confidences(
                confidences, boundary_list, half_patch_size, patch_area
            )
            
            total_confidence_time += time() - t_confidence_start
            
            t_data_start = time() # DEBUG
            boundary_data = get_data(
                im, unfilled, boundary_tuple, half_patch_size, patch_size)   
            total_data_time += time() - t_data_start # DEBUG
            
            target_point, target_confidence = get_priority_point(
                boundary_list, boundary_confidences, boundary_data
            )

            t_exemplar_start = time() # DEBUG
            source_patch = get_best_exemplar(
                target_point, im, original_mask, unfilled,  patch_size, compare_psz,
                distance_metric = distance_metric, restrict_search = restrict_search,
                euclidean_penalty=euclidean_penalty
            )
            total_exemplar_time += (time() - t_exemplar_start) # DEBUG
            
            fill_patch(
                im, target_point, source_patch, unfilled, confidences,
                target_confidence, half_patch_size, onion=onion
            )

        #Fill using onion layering
        else:
            while len(boundary_list) > 0:
                if show_movie:
                    cv.imshow('live progress', im)
                    cv.waitKey(5)
                if save_movie:    
                    #movie.append(np.copy(im))
                    movie.append(Image.fromarray((np.copy(im)*255).astype(np.uint8)))
                    
                target_point = boundary_list.pop()
                target_confidence = -1 #dummy value, since onion dosen't use confidence
            
                print("Target point: " + str(target_point))
                
                t_exemplar_start = time() # DEBUG
                source_patch = get_best_exemplar(
                    target_point, im, original_mask, unfilled,  patch_size, compare_psz,
                    distance_metric = distance_metric, restrict_search = restrict_search,
                    euclidean_penalty=euclidean_penalty
                )
                total_exemplar_time += (time() - t_exemplar_start) # DEBUG
                
                fill_patch(
                    im, target_point, source_patch, unfilled, confidences,
                    target_confidence, half_patch_size, onion=onion
                )
        
        
    total_time = time() - t_start # DEBUG

    
    print(
        "-----------------\n"
        "    TIMES (s)\n"
        "-----------------\n"
       f"Total:      {total_time:5.1f}\n"
       f"Exemplar:   {total_exemplar_time:5.1f}\n"
       f"Data:       {total_data_time:5.1f}\n"
       f"Confidence: {total_confidence_time:5.1f}\n"
        "-----------------\n"
    )

    
    if save_movie:#save gif
        imageio.mimsave("filling.gif", movie, duration = 20) 
    return im
    
