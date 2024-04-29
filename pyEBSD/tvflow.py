import numpy as np
from numpy.linalg import norm
from scipy.ndimage import gaussian_filter, convolve1d, convolve
from scipy.interpolate import griddata
from .orient import unit_quats
from warnings import warn

_DEBUG_OUTPUT = True
_DEBUG_OUTPUT_PERIOD = 10 # iterations

# TODO check with others that this value of epsilon is fine
_EPSILON = 1e-15 # arbitrary
_UNIT_VEC_NORM_TOLERANCE = 1e-10 # arbitrary

# TODO review
# TODO unit test
# For images with 2 spatial dimensions, "John Immerkaer, Fast Noise Variance
# Estimation. Computer Vision and Image Understanding" suggests using the kernel
#  1 -2  1
# -2  4 -2
#  1 -2  1
# which we separate into
#  1
# -2  x  1 -2  1
#  1
def _noise_variance_vector(x: np.ndarray) -> np.ndarray:
    """Estimates the noise variance of each channel of a given image"""
    # TODO numpy-style docstring
    # TODO what boundary condition should we use for the convolution?
    nchannels = x.shape[-1]
    variance_vector = np.empty(nchannels)
    npixels = np.prod(x.shape[:-1])
    n_spatial_dims = x.ndim - 1
    if n_spatial_dims == 2:
        kernel = [1, -2, 1]
        for channel in range(nchannels):
            temp  = convolve1d(x[..., channel], kernel, axis=0)
            noise = convolve1d(temp,            kernel, axis=1)
            # the Immerkaer paper divides by (36*(W-2)*(H-2)), presumably
            # because it doesn't include the border pixels in the convolution;
            # but we include the border pixels.
            variance_vector[channel] = \
                np.sum(noise**2, axis=(0, 1)) / (36*npixels)
    elif n_spatial_dims == 3:
        # TODO @make_volumetric implement the right kernel and division, then
        # remove this warning
        warn("Volumetric image noise estimation is implemented using "
             "placeholder values and is only suitable for testing purposes. "
             "Results should be considered arbitrary.")
        # TODO verify kernel correctness
        # TODO try to separate the kernel and do three 1-d convolutions instead
        kernel = [[[-6,   0, -6],
                   [ 0,  24,  0],
                   [-6,   0, -6]],

                  [[ 0,  24,  0],
                   [24, -96, 24],
                   [ 0,  24,  0]],

                  [[-6,   0, -6],
                   [ 0,  24,  0],
                   [-6,   0, -6]]]
        for channel in range(nchannels):
            # TODO when we get the actual kernel, if it is separable then
            # separate this into 3 1-dimensional convolutions for performance
            noise = convolve(x[..., channel], kernel)
            # TODO what do we divide by? 270 was chosen arbitrarily
            variance_vector[channel] = \
                np.sum(noise**2, axis=(0, 1, 2)) / (12960*npixels)
    else: raise ValueError("`x` must be a multichannel image with 2 or 3 "
                           "spatial dimensions.")
    return variance_vector

def _shift(x: np.ndarray, axis: int, direction: str) -> np.ndarray:
    # TODO docstring
    # TODO make clear in docstring that "+", a forward shift, means a shift
    # toward the end of the array; and "-", a backward shift, means a shift
    # toward the beginning of the array.
    # TODO should we check for valid axis, or let numpy handle that in indexing?

    shifted = np.copy(x)

    dim_len = x.shape[axis]
    forward_slices  = [slice(None)] * x.ndim
    backward_slices = [slice(None)] * x.ndim
    forward_slices[axis]  = slice(1, dim_len)
    backward_slices[axis] = slice(0, dim_len-1)
    forward_slices  = tuple(forward_slices)
    backward_slices = tuple(backward_slices)

    if   direction == "+": shifted[forward_slices]  = x[backward_slices]
    elif direction == "-": shifted[backward_slices] = x[forward_slices]
    else: raise ValueError("`direction` must be '+' or '-'.")

    return shifted

# TODO rename this function? it's not just a difference, it also imposes a
# boundary condition
def _diff(x: np.ndarray, axis: int, type: str) -> np.ndarray:
    # TODO docstring
    # TODO mention the imposed boundary condition in the docstring, incl. for
    # centered diff.
    if   type == "+": return _shift(x, axis, "-") - x
    elif type == "-": return x - _shift(x, axis, "+")
    elif type == "0": return (_shift(x, axis, "-") - _shift(x, axis, "+"))/2
    else: raise ValueError("`type` must be '+', '-', or '0'.")

# TODO review
# TODO unit test
def _kernel(q: np.ndarray, beta: float) -> np.ndarray:
    # TODO numpy-style docstring
    sigma = [0.7] * (q.ndim - 1) + [0] # we don't want to blur across channels
    q = gaussian_filter(q, sigma=sigma)

    squared_norm_grad = np.zeros(q.shape)
    for i in range(q.ndim - 1):
        # TODO the _diff function imposes a boundary condition. Is it correct to
        # use it here?
        squared_norm_grad += _diff(q, axis=i, type="0")**2

    alpha = 1 / np.sqrt(1 + squared_norm_grad/beta**2)
    return alpha

# TODO unit test
def _total_variation(q: np.ndarray, weight: np.ndarray=1) -> np.ndarray:
    squared_norm_grad = np.zeros(q.shape)
    for i in range(q.ndim - 1):
        squared_norm_grad += _diff(q, axis=i, type="0")**2
    return np.sum(
        weight * np.sqrt(squared_norm_grad),
        axis=tuple(range(q.ndim - 1))
    )

# expects a multichannel image, even if the number of channels is 1 (i.e. the
# last dimension must be the one separating channels, even if it is 1 element in
# length)
def _fill_missing(x: np.ndarray):
    for channel in range(x.shape[-1]):
        ch_slice = x[..., channel]
        nan_mask = np.isnan(ch_slice)
        missing_coords = np.nonzero(nan_mask)
        known_coords = np.nonzero(~nan_mask)
        ch_slice[missing_coords] = griddata(known_coords, ch_slice[known_coords],
                                            missing_coords, method="nearest")
        # TODO want to use "method=linear" instead, but it doesn't interpolate
        # over some NaN values -- maybe because they are outside the convex hull
        # of the known points

# TODO should we be weighted by default?
# TODO @make_volumetric update docstring
def denoise(q: np.ndarray, max_iters: int=5000, dt: float=0.005,
            force_max_iters: bool=False,
            weighted: bool=False, beta: float=0.0005,
            on_quats: bool=True,
            weight_array: np.ndarray=None) -> np.ndarray:
    """Denoise quaternion orienation map using TV flow.

    Performs iterative denoising on the quaternion image `q` using total
    variation flow, where `q` has 2 spatial dimensions and each pixel represents
    a quaternion. All quaternions in `q` are normalized before denoising.

    Parameters
    ----------
    q : numpy.ndarray
        An array of shape (M, N, 4), where each `q[m, n, :]` represents a
        quaternion.
        Automatically gets normalized to unit quaternions.
        Must not contain NaN values.
    max_iters : int, default=5000
        The function automatically determines how many iterations to perform,
        but will never exceed `max_iters`.
    dt : float, default=0.005
        The time step to use between iterations.
    weighted : bool, default=False
        When true, computes a weighting for the TV flow to more strongly
        preserve edges. This causes less blurring, but requires many more
        iterations.
    beta : float, default=0.0005
        TODO how should we describe this?

    Returns
    -------
    numpy.ndarray
        Denoised copy of `q`, normalized to unit quaternions.

    Raises
    ------
    ValueError
        If `q` is not a 3-dimensional array.
        If `q` contains NaN values.

    Warns
    -----
    UserWarning
        If `q` contains non-unit quaternions.
        If `max_iters` is reached.
    """

    return _tvflow_operation(
        operation="denoise",
        q=q,
        max_iters=max_iters,
        force_max_iters=force_max_iters,
        dt=dt,
        weighted=weighted,
        beta=beta,
        on_quats=on_quats,
        weight_array=weight_array
    )
# TODO @make_volumetric update docstring
def inpaint(q: np.ndarray, max_iters: int=500000, dt: float=0.005,
            force_max_iters: bool=False,
            delta_tolerance: float=1e-5,
            on_quats: bool=True,
            weight_array: np.ndarray=None) -> np.ndarray:
    """Inpaint quaternion orienation map using TV flow.

    Performs iterative inpainting on the quaternion image `q` using total
    variation flow, where `q` has 2 spatial dimensions and each pixel represents
    a quaternion, and missing pixels are represented by `[NaN, NaN, NaN, NaN]`.
    All quaternions in `q` are normalized before inpainting.

    Parameters
    ----------
    q : numpy.ndarray
        An array of shape (M, N, 4), where each `q[m, n, :]` represents a
        quaternion.
        Automatically gets normalized to unit quaternions.
        Must not contain NaN values.
    max_iters : int, default=5000
        The number of iterations is determined by `delta_tolerance`, but will
        never exceed `max_iters`.
    dt : float, default=0.005
        The time step to use between iterations.
    delta_tolerance : float, default=1e-5
        The algorithm stops when the change in the image over an iteration is
        very small; 'how small' is determined by `delta_tolerance`. I.e., a
        larger `delta_tolerance` stops the algorithm after fewer iterations.
        Specifically, the algorithm stops inpainting each channel when the
        Frobenius norm of the change in that channel over an iteration is less
        than `delta_tolerance*dt`.

    Returns
    -------
    numpy.ndarray
        Inpainted copy of `q`, normalized to unit quaternions.

    Raises
    ------
    ValueError
        If `q` is not a 3-dimensional array.

    Warns
    -----
    UserWarning
        If `q` contains non-unit quaternions.
        If `max_iters` is reached.
    """

    return _tvflow_operation(
        operation="inpaint",
        q=q,
        max_iters=max_iters,
        force_max_iters=force_max_iters,
        dt=dt,
        delta_tolerance=delta_tolerance,
        on_quats=on_quats,
        weight_array=weight_array
    )

# TODO there's no error checking
def _get_denoising_active_channels(q: np.ndarray, q0: np.ndarray,
                                   noise_variance: np.ndarray,
                                   n_active_pixels: int):
    variance_removed_noise = (
        np.sum(
            (q0 - q)**2,
            axis=tuple(range(q.ndim - 1))
        ) / n_active_pixels
    )
    return variance_removed_noise <= noise_variance
# TODO there's no error checking
def _get_inpainting_active_channels(q: np.ndarray, q_prev: np.ndarray,
                                    delta_tolerance: float,
                                    n_active_pixels: int):
    frobenius_norms_delta_per_nan_pixel = (
        np.sqrt(
            np.sum(
                (q_prev - q)**2,
                axis=tuple(range(q.ndim - 1))
            )
        ) / n_active_pixels
    )
    q_prev[...] = q
    return frobenius_norms_delta_per_nan_pixel > delta_tolerance

# Implementation note: ndarray quantities which are different depending on
# spatial dimension (e.g. gradient) are stored in a single ndarray, in which the
# first axis separates the ndarrays for different spatial dimensions. E.g., let
# ndarrays g0, g1, and g2 be the gradient components of a volumetric image,
# where gn is the gradient component in spatial dimension n. Each gn has the
# same array dimensions as the image array, and they are stored together in an
# ndarray [g0, g1, g2].
def _tvflow_operation(operation: str, q: np.ndarray, max_iters: int, dt: float,
                      force_max_iters: bool,
                      weighted: bool=None, beta: float=None, # denoise only
                      delta_tolerance: float=None,           # inpaint only
                      on_quats: bool=True,
                      weight_array: np.ndarray=None
                     ) -> np.ndarray:
    """Perform 2D TV flow denoising or inpainting.

    Denoises or inpaints `q` depending on the value of `operation`.
    For internal use only; users of this module should use the `denoise2d` and
    `inpaint2d` wrapper functions.

    Parameters
    ----------
    operation : {'denoise', 'inpaint'}

    Raises
    ------
    TypeError
        If operation is 'denoise' and `weighted` or `beta` are not given.
        If operation is 'inpaint' and `delta_tolerance` is not given.

    See also
    --------
    `denoise2d` and `inpaint2d` wrapper function docstrings for details.
    """

    q = q.copy()
    if q.ndim not in (3, 4):
        raise ValueError("`q` must be a 3- or 4-dimensional array.")
    if on_quats and np.any(np.abs(norm(q, axis=-1) - 1) > _UNIT_VEC_NORM_TOLERANCE):
        warn("`q` contains non-unit quaternions. Normalizing.")
        q = unit_quats(q)
    weight = 1
    weight_forward = 1
    if operation == "denoise":
        if weighted is None or beta is None:
            raise TypeError("Parameters `weighted` and `beta` must be "
                            "specified when denosing.")
        if np.any(np.isnan(q)):
            raise ValueError("`q` must not contain NaN values.")
        constant_mask = False
        if weighted:
            if weight_array is None:
                weight = _kernel(q, beta)
            else:
                weight = np.dstack((weight_array,weight_array,weight_array))
            weight_forward = [None] * (q.ndim - 1)
            for axis in range(q.ndim - 1):
                weight_forward[axis] = _shift(weight, axis, "+")
            weight_forward = np.array(weight_forward)
        n_active_pixels = np.prod(q.shape[:-1])
        threshold = _noise_variance_vector(q)
        get_active_channels = _get_denoising_active_channels
        q0 = np.copy(q)
        q_compare = q0
    elif operation == "inpaint":
        if delta_tolerance is None:
            raise TypeError("Parameter `delta_tolerance` must be specified "
                            "when inpainting.")
        nan_mask = np.isnan(q)
        # we assume all channels have the same nan pixels
        n_active_pixels = np.sum(nan_mask[..., 0])
        if n_active_pixels == 0: warn("`q` does not contain missing pixels.")
        constant_mask = ~nan_mask
        threshold = dt*delta_tolerance
        get_active_channels = _get_inpainting_active_channels
        _fill_missing(q)
        q0 = np.copy(q)
        q_compare = np.copy(q)
    else: raise ValueError("Invalid value for `operation`.")

    if _DEBUG_OUTPUT:
        debug_counter = 0
        prev_ch_active = np.array([True] * q.shape[-1])
    # TODO see if you can avoid performing unnecessary calculations on inactive
    # channels (only if it actually improves performance)
    ch_active = np.array([True] * q.shape[-1])
    for iter in range(max_iters):
        q_forward  = [None] * (q.ndim - 1)
        q_backward = [None] * (q.ndim - 1)
        for axis in range(q.ndim - 1):
            q_forward [axis] = _shift(q, axis, "+")
            q_backward[axis] = _shift(q, axis, "-")
        q_forward  = np.array(q_forward)
        q_backward = np.array(q_backward)

        vec_tv = _total_variation(q, weight=weight)
        R = vec_tv / (_EPSILON + norm(vec_tv))
        C = [None] * (q.ndim - 1)
        for axis in range(q.ndim - 1):
            centered_diff_axes = list(range(q.ndim - 1))
            centered_diff_axes.remove(axis)
            sum_sq_centered_diffs = np.zeros(q.shape)
            for centered_diff_axis in centered_diff_axes:
                sum_sq_centered_diffs += _diff(q, centered_diff_axis, "0")**2
            C[axis] = 1 / np.sqrt(
                _EPSILON + _diff(q, axis, "+")**2 + sum_sq_centered_diffs
            )
        C = np.array(C)
        C_forward = [None] * (q.ndim - 1)
        for axis in range(q.ndim - 1):
            C_forward[axis] = _shift(C[axis], axis, "+")
        C_forward = np.array(C_forward)
        C_dot_q = weight * np.sum(C * q_backward, axis=0) + \
                  np.sum(weight_forward * C_forward * q_forward, axis=0)
        sum_C = weight * np.sum(C, axis=0) + \
                np.sum(weight_forward * C_forward, axis=0)

        q[..., ch_active] += R[ch_active] * dt * C_dot_q[..., ch_active]
        q[..., ch_active] /= 1 + R[ch_active] * dt * sum_C[..., ch_active]

        q[constant_mask] = q0[constant_mask]
        if not force_max_iters:
            ch_active = get_active_channels(
                q, q_compare, threshold, n_active_pixels
            )

        if _DEBUG_OUTPUT:
            if debug_counter == 0:
#                print(f"{iter:{len(str(max_iters))}}: {ch_active}", end="\r")
                debug_counter = _DEBUG_OUTPUT_PERIOD
            if (ch_active != prev_ch_active).any():
#                print(f"{iter:{len(str(max_iters))}}: {ch_active}")
                prev_ch_active = ch_active
            debug_counter -= 1

        if not np.any(ch_active):
            break
    if iter >= max_iters - 1:
        warn(f"Reached maximum number of iterations ({max_iters}).")

    if on_quats:
        q = unit_quats(q)
        # `unit_quats` very slightly changes known values, so:
        q[constant_mask] = q0[constant_mask]

    return q
