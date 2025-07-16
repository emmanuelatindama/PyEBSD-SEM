from sys import path
path.insert(0, "..") # hack to get module `pyEBSD` in scope
import pyEBSD as ebsd

import os
import time
from time import time  # DEBUG

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from scipy.spatial import cKDTree # Used for nearest neighbor inpainting
from scipy import ndimage

import imageio #For the gif
from PIL import Image



def nearest_neighbor_filling(image, mask):
    """
    Fills the masked region of an image using nearest neighbor interpolation.
    
    Parameters:
    - image: The input image with masked regions (numpy array).
    - mask: The binary mask indicating the masked regions (numpy array).

    Returns:
    - inpainted_image: The image with the masked regions filled (numpy array).
    """
    inpainted_image = image.copy()
    
    # Find the coordinates of the known and unknown (masked) pixels
    known_coords = np.array(np.nonzero(~mask)).T
    unknown_coords = np.array(np.nonzero(mask)).T
    
    # Create a KDTree for the known pixel coordinates
    tree = cKDTree(known_coords)
    
    # For each unknown pixel, find the nearest known pixel and copy its value
    for unknown_coord in unknown_coords:
        dist, index = tree.query(unknown_coord)
        nearest_known_coord = known_coords[index]
        inpainted_image[tuple(unknown_coord)] = image[tuple(nearest_known_coord)]
    
    return inpainted_image


def apply_transformations(patch):
    """
    Generate all possible transformations (rotations and flips) of a patch.
    
    Parameters:
    -----------
    patch : np.ndarray
        The input patch.
        
    Returns:
    --------
    list of np.ndarray
        A list of transformed patches.
    """
    transforms = [patch, np.rot90(patch, 1), np.rot90(patch, 2), np.rot90(patch, 3), np.fliplr(patch),
                  np.flipud(patch), np.fliplr(np.rot90(patch, 1)), np.flipud(np.rot90(patch, 1))]
    return transforms

def min_source_region(image, mask, patch_size):
    source_space = []
    half_patch_size = patch_size // 2
    source_image = image.copy(); source_image[mask] = 0
    for i in range(source_image.shape[0]):
        for j in range(source_image.shape[1]):
            point_patch = patch(source_image, (i,j), half_patch_size)
            if np.all(np.sum(point_patch, axis=2)):
                source_space.append(point_patch)  
    return source_space

def create_source_region(image, mask, patch_size):
    source_space = []
    half_patch_size = patch_size // 2
    source_image = image.copy(); source_image[mask] = 0
    for i in range(source_image.shape[0]):
        for j in range(source_image.shape[1]):
            point_patch = patch(source_image, (i,j), half_patch_size)
            if np.all(np.sum(point_patch, axis=2)):
                source_space.extend(apply_transformations(point_patch))   
    return source_space


def get_boundary(unfilled_mask: np.ndarray, patch_size: int) -> np.ndarray:
    """ get_boundary
    A boolean np.array that is the shape of the inputted image. This array is all false except on the
    boundary points, which are true
    
    Inputs:
        unfilled_mask: An np.array. The mask of image
        patch_size: int. This is used to account for the padding in the mask

    Returns:
        returns the boundary of known pixels around the masked region
        
    """
    boundary = ndimage.binary_dilation(unfilled_mask, np.full((3, 3), True)) & ~unfilled_mask
    
    # Exclude edges from boundary.
    boundary[:patch_size//2, :] = 0
    boundary[-(patch_size//2):, :] = 0
    boundary[:, :patch_size//2] = 0
    boundary[:, -(patch_size//2):] = 0
    return boundary


def patch_slice(point: tuple, half_patch_size: int) -> tuple:
    """patch_slice
    Inputs:
        point: a tuple. the index of the center point that we want the patch of
        half_patch_size: and int. the size of the desired patch divided by 2. 
    Returns:
        A tuple that contains all four corner indexes of the patch.
    """
    return (slice(max(point[0] - half_patch_size, 0), point[0] + half_patch_size+1),
            slice(max(point[1] - half_patch_size, 0), point[1] + half_patch_size+1))


def patch(image: np.ndarray, point: tuple, half_patch_size: int):
    """patch
    Inputs:
        image: np.ndarray, this is the array that we will pull the patch from. 
            i.e. it could pull that patch from the image we are filling in
                or pull the patch from the working mask so we know what is valid.
        point: A tuple. This is the center point of the patch that we want. 
        half_patch_size: an int, this is the size of the patch that we want. 
    Returns:
        The patch centered at the point in the image
    """
    return image[patch_slice(point, half_patch_size)]


def get_confidence(prev_confidences: np.ndarray, point: tuple, half_patch_size: int, patch_area: int) -> float:
    """ Confidence term represents the amount of reliable information surrounding a pixel/point.
    Inputs: 
        prev_confidences: An ndarray, which is the array of the confidences for
            all points in the image. Note: expects "prev_confidences" to be 0 in the unfilled region.
        point: a tuple, this is the point that we want to get the confidence of. 
        half_patch_size: an int, this is the size of the patch we are using /2.
        patch_area: an int, this is the area of the patch we are using
    Returns
        The sum of all the confidences in the patch, which will then become 
            the confidence of the inputted point,
    """
    return np.sum(patch(prev_confidences, point, half_patch_size))/ patch_area


def get_confidences(prev_confidences: np.ndarray, boundary: list, half_patch_size: int, patch_area: int) -> list:
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


def calc_normal_matrix(in_mask: np.ndarray, boundary_list: list) -> np.ndarray:
    """ Calculate the normal matrix for a given mask and boundary.
        Uses a sobel matrix to find the boundary of the mask. Calculates the strength of
        isophotes throughout the mask

    Parameters:
    -----------
    in_mask : np.ndarray
        The working mask of the image.
    boundary_list : list
        A list of tuples representing the boundary points around the unfilled regions.
    
    Returns:
    --------
    np.ndarray
        A matrix that contains the normal vectors at each boundary point in the mask.
    """
    # Compute gradients using Sobel filter
    gradient_x = ndimage.sobel(in_mask.astype(np.float32), axis=1)
    gradient_y = ndimage.sobel(in_mask.astype(np.float32), axis=0)
    
    normals_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Compute unit normal vectors
    unit_normals = np.zeros((in_mask.shape[0], in_mask.shape[1], 2), dtype=np.float32)
    for point in boundary_list:
        row, col = point
        
        # Compute the normal vector (perpendicular to the gradient) We do not do this np.array([-gy, gx])
        # since the finite difference computation of gradients give us vectors perpendicular to the boundary
        unit_normals[row, col, 1] = gradient_y[row, col]
        unit_normals[row, col, 0] = gradient_x[row, col]
        
        # Normalize the gradient vector to unit vectors
        if normals_magnitude[row, col] != 0:
            unit_normals[row, col] /= normals_magnitude[row, col]
                
    return unit_normals


def calc_gradient(image: np.ndarray, boundary_list: list) -> np.ndarray:
    """
    Calculate the gradient of an image, taking into account the unfilled regions and boundary points.

    Parameters:
    -----------
    image : np.ndarray
        The input image for which the gradient is to be calculated. Expected to be in RGB format.
    boundary_list : list
        A list of tuples representing the boundary points around the unfilled regions.
    
    Returns:
    --------
    np.ndarray
        An array of the same height and width as the input image, with two channels representing the maximum gradient values in the x and y directions at each boundary point.

    Notes:
    ------
    - The function first converts the input image to grayscale and computes the gradients in the x and y directions.
    - For each boundary point, the function calculates the gradient magnitudes within a patch around the point and finds the maximum gradient value in that patch.
    - The gradient values in the x and y directions are returned in the 1st and 2nd channels of a 2-channel array.
    
    Example:
    --------
    >>> image = np.random.rand(100, 100, 3)
    >>> boundary_list = [(10, 10), (20, 20), (30, 30)]
    >>> result = calc_gradient(image, boundary_list)
    >>> print(result.shape)
    (100, 100, 2)
    """
    # Scale the image without losing the information in any of the channels (more weight to middle channel)
    grayscale_image = np.dot(image[...,:],[0.299, 0.587, 0.114])
    
    # Compute gradients using Sobel filter
    gradient_x, gradient_y = ndimage.sobel(grayscale_image, axis=1), ndimage.sobel(grayscale_image, axis=0)
    
    height, width = image.shape[:2]
    
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    unit_gradient = np.zeros((image.shape[0], image.shape[1], 2), dtype=np.float32)
    
    for point in boundary_list:
        row, col = point
        unit_gradient[row, col, 1] = gradient_y[row, col]
        unit_gradient[row, col, 0] = gradient_x[row, col]
        
        # Normalize the gradient vector to unit vectors
        if gradient_magnitude[row, col] != 0:
            unit_gradient[row, col] /= gradient_magnitude[row, col]
    
    return unit_gradient


def get_data(original_image: np.ndarray, working_mask: np.ndarray, boundary_list: list, output=None):
    """ Calculate the data term for the Criminisi inpainting algorithm.
        Data term: Indicates the strength of the isophote (contour line) hitting the boundary, which helps
        to propagate the structure. It is computed as |dot_product(gradient_perp,unit_normal)|/normalizing_factor.
        We use the original image instead of the working image because, we do not want to propagate any potential
        erroroneous values that were used to fill the missing region in the previous iteration.
        Meanwhile we trust our original image since it is the result of an ML prediction/nearest neighbor filling,
        and so will have have either reliable or null effects.
        
    Parameters:
    -----------
    original_image : np.ndarray
        The  image with shape (H, W, 3).
    working_mask : np.ndarray
        The mask with shape (H, W).
    boundary_list : np.ndarray
        The binary array indicating the boundary of the mask.
    Returns:
    --------
    np.ndarray
        An array of the same height and width as the input image, representing the data term values at each boundary point.
    """
    normals = calc_normal_matrix(working_mask, boundary_list)
    
    # Dx = gradient[:, :, 0] and Dy = gradient[:, :, 1]
    gradient = calc_gradient(original_image, boundary_list)
    gradient_perp = np.dstack((-gradient[:, :, 1], gradient[:, :, 0]))
     
    data = np.abs(np.sum(gradient_perp*normals, axis=2))
    
    data_array = np.zeros_like(normals[:,:,0])
    for point in boundary_list:
        data_array[point] = data[point] + 0.001
        
    if output:
        return [data[point] for point in boundary_list]
    return data_array


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


def SSE(target_patch, source_patch, mask, known_weight=1.0):
    """
    Calculate the distance between the target patch and a source patch with weights for known and unknown regions.
    
    Parameters:
    -----------
    target_patch : np.ndarray
        The target patch with some missing regions.
    source_patch : np.ndarray
        A source patch from the image.
    mask : np.ndarray
        The mask indicating the missing regions in the target patch.
    known_weight : float
        The weight for the known regions in the target patch.
    
    Returns:
    --------
    float
        The distance between the patches.
    """
    known_region = (1 - mask)
    unknown_region = mask
    
    if target_patch.shape == source_patch.shape:
        if target_patch.ndim == 3:
            diff = np.sum((target_patch - source_patch) ** 2, axis=2)
        else:
            diff = (target_patch - source_patch) ** 2
        
        weighted_diff = known_weight * diff * known_region + (1-known_weight) * diff * unknown_region

        distance = np.sum(weighted_diff)
    else:
        distance = 1000000.0
    return distance


def MISORI(target_patch, source_patch, mask, known_weight=1.0):
    """
    Calculate the Misorientation between the target patch and a source patch with weights for known and unknown regions.
    
    Parameters:
    -----------
    target_patch : np.ndarray
        The target patch with some missing regions.
    source_patch : np.ndarray
        A source patch from the image.
    mask : np.ndarray
        The mask indicating the missing regions in the target patch.
    known_weight : float
        The weight for the known regions in the target patch.
    
    Returns:
    --------
    float
        The distance between the patches.
    """
    known_region = (1 - mask)
    unknown_region = mask
    
    if target_patch.shape == source_patch.shape:
        if target_patch.ndim == 3:
            diff = ebsd.orientation.misorientation_mat(target_patch, source_patch, symmetry_op = 'hexagonal')
        weighted_diff = known_weight * diff * known_region + (1-known_weight) * diff * unknown_region
        distance = np.sum(weighted_diff)
    else:
        distance = 1000000.0
    return distance


def get_best_exemplar(original_image, target_point, patch_size, source_region, working_mask, loss, known_weight=1.0):
    """
    Find the best patch in the image to inpaint the region around the target point.
    
    Parameters:
    -----------
    image : np.ndarray
        The image.
    target_point : tuple
        The coordinates (y, x) of the center of the target patch.
    patch_size : int
        The size of the patch (must be odd).
    mask : np.ndarray
        The binary mask indicating the missing regions in the target patch.
    known_weight : float
        The weight for the known regions in the target patch.
    loss : str
    	choose between sse and delta (disorientation angle).
     
    Returns:
    --------
    tuple
        The coordinates (y, x) of the best matching patch in the source image.
        The best matching patch itself.
    """
    half_patch_size = patch_size // 2; target_y, target_x = target_point
    
    target_patch = patch(original_image, target_point, half_patch_size)
    mask_patch = patch(working_mask, target_point, half_patch_size)
    
    best_patch = None; min_distance = float('inf')
    
    for source_patch in source_region:
        if loss == 'delta':
            distance = MISORI(target_patch, source_patch, mask_patch, known_weight)
        else:
            distance = SSE(target_patch*2*np.pi, source_patch*2*np.pi, mask_patch, known_weight)
       
        #distance = MISORI(target_patch, source_patch, mask_patch, known_weight) #SSE(target_patch, source_patch, mask_patch, known_weight)
        if distance < min_distance:
            min_distance = distance
            best_patch = source_patch
        if min_distance < 0.005:
            return target_patch, best_patch, mask_patch

    return target_patch, best_patch, mask_patch


def fill_patch(working_image, target_point, best_patch, mask_patch,
               working_mask, confidences, target_confidence, half_psz):
    """fill_patch
    Inputs:
        working_image: np.ndarray, the working image
        target_point: tuple, the point we are filling in
        mask_patch
        best_patch: np.ndarray, the patch we are using for filling
        working_mask: np.ndarray, the working mask indicating what is and isnt filled in
        confidences: np.ndarray, array indicating the confidences of all pixels in the image
        target_confidence: float,the confidence of the target pixel
        half_patch_size: int, the size of the patch used for filling/2
    Does:
        Updates the unfilled image, working mask and confidences
    Note:
        modifies some inputs in-place
    """
    
    target_patch_slice = patch_slice(target_point, half_psz)
    
    # Update confidence
    confidences[target_patch_slice][mask_patch] = target_confidence
    
    # Update image with the unfilled_patch of the best_patch
    working_image[target_patch_slice][mask_patch] = best_patch[mask_patch]
    
    # Update mask 
    working_mask[target_patch_slice][mask_patch] = False
    

def inpaint(image, mask, psz=3, using_ml_template:bool = False, learn_from_algorithm:bool = False,
            save_movie:bool = True, movie_path=None, known_weight:int = 1, plot_progress:bool = False, loss:str='sse', ST:bool = False):
    """
    Perform image inpainting using an exemplar-based algorithm.

    This function fills in missing regions of an image using a patch-based 
    inpainting approach. The inpainting is guided by patch similarity and 
    confidence measures, allowing for seamless restoration of damaged areas.

    Parameters:
    ----------
    image : ndarray
        The input image with missing regions to be inpainted. Should be a 3D array 
        (height, width, channels).
    mask : ndarray
        A binary mask indicating the missing regions to be inpainted. Should be a 2D 
        array (height, width) where 1 represents missing pixels and 0 represents 
        known pixels.
    psz : int
        The size of the patch used for inpainting. Should be an odd integer.
    using_ml_template : bool, optional
        If True, use a machine learning-based template for inpainting. Default is False.
    learn_from_algorithm : bool, optional
        If True, allow the algorithm to learn from its own inpainting results. Default is False.
    save_movie : bool, optional
        If True, save the inpainting process as a GIF. Default is True.
    known_weight : int, optional
        The weight given to known pixels when computing patch similarity. Default is 0.99.
    plot_progress : bool, optional
        If True, plot the progress of the inpainting process. Default is False.
    loss : str
    	choose between sse and delta (disorientation angle).
    ST : bool
    	If True, use source transform
    Returns:
    -------
    confidences : ndarray
        The final confidence map after inpainting.
    working_image : ndarray
        The inpainted image with the same dimensions as the input image.

    Notes:
    -----
    - The function pads the image and mask to handle edge cases during patch processing.
    - The inpainting process continues until all missing regions are filled.
    - The function supports optional plotting and GIF saving to visualize the inpainting progress.

    Example:
    -------
    >>> image = cv2.imread('image_with_hole.jpg')
    >>> mask = cv2.imread('mask.jpg', 0)
    >>> mask = mask // 255  # Convert to binary mask
    >>> confidences, inpainted_image = inpaint(image, mask, psz=9, save_movie=True)
    >>> cv2.imwrite('inpainted_image.jpg', inpainted_image)
    """
    half_psz = psz // 2
    patch_area = psz**2
    
    loss = loss.lower()
    if mask.ndim == 3:
        mask = mask[:,:,0]
    
    mask_padding = (half_psz, half_psz)
    padded_mask = np.pad(mask, mask_padding, mode='edge')
    original_mask = padded_mask.copy()
    working_mask = padded_mask.copy()
    
    image_padding = ((half_psz,), (half_psz,), (0,))
    padded_image = np.pad(image, image_padding, mode='edge')
    original_image = padded_image.copy()
    
    if using_ml_template:
        working_image = padded_image.copy()
    else:
        working_image = nearest_neighbor_filling(original_image, original_mask)
    
    if ST:
    	source_region = create_source_region(original_image, original_mask, psz) # This is fixed
    else:
    	source_region = min_source_region(original_image, original_mask, psz) # This is fixed
    "Initate confidence. C(known) = 1 and C(unknown) = 0"
    confidences = (~original_mask).copy().astype(float) # Initialized, but changes in each iteration
    
    #Used for Gif of filling
    if save_movie:
        movie = []
    iter_counter = 0
    
    total_exemplar_time = 0 # DEBUG
    total_data_time = 0 # DEBUG
    total_confidence_time = 0 # DEBUG
    t_start = time() # Overall Start Time - DEBUG
    iter_counter = 0 # DEBUG

    #Main loop, continue filling until nothing left to fill
    while np.sum(working_mask[half_psz : -half_psz, half_psz: -half_psz])>0:
        iter_counter += 1
        # print(f"\nStarting iteration: {iter_counter}")
        
        if save_movie:
            gif_image = working_image.copy(); gif_image[working_mask] = 1
            movie.append(Image.fromarray((gif_image*255).astype(np.uint8)))
        
        "retrieve boundary using working mask because we update it in each iteration"
        boundary_array = get_boundary(working_mask, patch_size=psz)
        boundary_tuple = boundary_array.nonzero()
        boundary_list = list(zip(boundary_tuple[0], boundary_tuple[1]))
        t_confidence_start = time() # Start Time - DEBUG
        "confidence has been initialized, and so we only select boundary confidence values"
        boundary_confidence_list = get_confidences(confidences, boundary_list, half_psz, patch_area)
        total_confidence_time += time() - t_confidence_start # End Time - DEBUG
        
        t_data_start = time() # Start Time - DEBUG
        if using_ml_template:
            "We use original_image to compute data_term under the assumption that the input is the ML prediction"
            data_list = get_data(original_image, working_mask, boundary_list, output = list)
        else:
            data_list = get_data(working_image, working_mask, boundary_list, output = list)
        total_data_time += time() - t_data_start # End Time - DEBUG
            
        "Use confidence and data terms to compute priority term along the boundary"
        target_point, target_confidence = get_priority_point(boundary_list, boundary_confidence_list, data_list)
        #print("Target point: " + str(target_point))
        
        t_exemplar_start = time() # Start Time - DEBUG
        if learn_from_algorithm:
            target_patch, best_patch, mask_patch = get_best_exemplar(working_image, target_point, psz, source_region, working_mask, known_weight=known_weight, loss=loss)
        else:
            target_patch, best_patch, mask_patch = get_best_exemplar(original_image, target_point, psz, source_region, working_mask, known_weight=known_weight, loss=loss)
        total_exemplar_time += (time() - t_exemplar_start) #End Time - DEBUG
              
        fill_patch(working_image, target_point, best_patch, mask_patch, working_mask, confidences, target_confidence, half_psz)
        
        if plot_progress:
            plt.figure(figsize=(15,10))
            gif_image = working_image.copy(); gif_image[working_mask] = 1
            plt.subplot(2,3,4); plt.imshow(confidences); plt.axis('off'); plt.title('Updated Confidence')
            plt.subplot(2,3,5); plt.imshow(gif_image); plt.axis('off'); plt.title('Updated Image')
            plt.subplot(2,3,6); plt.imshow(working_mask, cmap='gray'); plt.axis('off'); plt.title('Updated Mask')
            plt.show()
    total_time = time() - t_start # End Time - DEBUG    
    
    if save_movie: #save gif
        gif_image = working_image.copy(); gif_image[working_mask] = 1
        movie.append(Image.fromarray((gif_image*255).astype(np.uint8)))
        # print(len(movie))
        if movie_path:
            imageio.mimsave(movie_path+".gif", movie, duration = 0.05)
        else:
            imageio.mimsave("exemplar.gif", movie, duration = 0.05)
    # print(
    #     "-----------------\n"
    #     "    TIMES (s)\n"
    #     "-----------------\n"
    #    f"Total:      {total_time:5.1f}\n"
    #    f"Exemplar:   {total_exemplar_time:5.1f}\n"
    #    f"Data:       {total_data_time:5.1f}\n"
    #    f"Confidence: {total_confidence_time:5.1f}\n"
    #     "-----------------\n"
    # )
        
    return working_image[half_psz : -half_psz, half_psz : -half_psz]
