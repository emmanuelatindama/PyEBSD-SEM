import numpy as np
from scipy.spatial.transform import Rotation as R


'''
Original author: Arthur Nishikawa
Modified by: Emmanuel Atindama

pyebsd is an python implemented open-source tool for processing Electron Backscattered Diffraction (EBSD) data. The main implemented features are:

We employ intrinsic rotations in this entire implementation. That is, the new cordinate system after each rotation is the rotated cordinate system.
Intrinsic rotations are also known as passive rotations. INTRINSIC ROTATION is denoted using UPPER CASE LETTERS vs extrinsic rotation which uses lower case letters.

pole figures
inverse pole figures for cubic crystals
accurate orientation relationship for cubic crystals
misorientation
pyebsd is in development stage and, for now, is only able to process phases of cubic and hexagonal symmetry. Its main use case is steel and titanium microstructures.

pyebsd is a "pure" Python package, meaning that it does not depend on building extension modules. For computational expensive calculations, such as manipulation of matrices, it relies on the clever vectorized operations with NumPy.


'''

__all__ = ['mean_misorientation_ang', 'misorientation_euler', 'misorientation',
           'cubic_symmetry_operators','hexagonal_symmetry_operators', 'rotation_matrix_to_axis_angle','axis_angle_to_rotation_matrix']

# def trace_to_angle(tr, out='deg'):
#     """
#     Converts the trace of a rotation/orientation matrix to the misorientation angle
#     """
#     ang = np.arccos((tr-1.)/2.)
#     if out == 'deg':
#         ang = np.degrees(ang)
#     return ang

def rtmean_sq_misorientation_ang(arr1, arr2, symmetry_op = 'cubic'):
    """
    Author: Emmanuel Atoleya Atindama (EAA)
    Computes the mean square misorientation between two arrays containing Euler angles arr1, and arr2, 
    with [theta1, Theta, theta2] at each [i,j,:] of arrays 1 and 2.
    Array shapes are nxnx3. Each 1x1x3 contains 1 euler angle
    
    Parameters
    ----------
    e1 : numpy ndarray shape(n,n,3)
        First rotation matrix
    e1 : numpy ndarray shape(n,n,3)
        Second rotation matrix
    Returns
    -------
    misang : float
        Misorientation ang
    """
    misori = 0
    for i in range(arr1.shape[0]):
        for j in range(arr1.shape[1]):
            misori += misorientation_euler(arr1[i,j], arr2[i,j], symmetry_op = symmetry_op)**2
    
    return np.sqrt(misori / (arr1.shape[0]*arr1.shape[1]))



def mean_misorientation_ang(arr1, arr2, symmetry_op = 'cubic'):
    """
    Author: Emmanuel Atoleya Atindama (EAA)
    Computes the average misorientation between two arrays containing Euler angles arr1, and arr2, 
    with [theta1, Theta, theta2] at each [i,j,:] of arrays 1 and 2.
    Array shapes are nxnx3. Each 1x1x3 contains 1 euler angle
    
    Parameters
    ----------
    e1 : numpy ndarray shape(n,n,3)
        First rotation matrix
   e1 : numpy ndarray shape(n,n,3)
        Second rotation matrix
    Returns
    -------
    misang : float
        Misorientation ang
    """
    misori = 0
    for i in range(arr1.shape[0]):
        for j in range(arr1.shape[1]):
            misori += misorientation_euler(arr1[i,j], arr2[i,j], symmetry_op = symmetry_op) 
    return misori/(arr1.shape[0]*arr1.shape[1])

def misorientation_euler(e1, e2, out = 'rad', symmetry_op = 'cubic'):
    """
    Author: Emmanuel Atoleya Atindama (EAA)
    Computes the misorientation between two Euler angles e1=[phi1, Phi, phi2], e2=[theta1, Theta, theta2]
    
    Parameters
    ----------
    e1 : numpy ndarray shape(3,1)
        First rotation matrix
   e1 : numpy ndarray shape(3, 1)
        Second rotation matrix
    out : str (optional)
        'deg': as misorientation angle in degrees
        'rad': as misorientation angle in radians
        Default: 'deg'
    symmetry_op : str (optional)
        Crystal symmetry of the material:
        'cubic': cubic symmetry has 24 symmetries
        'hexagonal': hexagonal symmetry has 12 symmetries
        Default: 'cubic'
    
    Returns
    -------
    misang : float
        Misorientation ang
    """
    e1 = np.array(e1); e2 = np.array(e2)
    # Check whether Euler angles are in degrees or radian
    e1 = np.deg2rad(e1) if np.max(e1) > 2*np.pi else e1 # convert to radians if data in degrees
    e2 = np.deg2rad(e2) if np.max(e2) > 2*np.pi else e2 # convert to radians if data in degrees
    
    
    r1 = R.from_euler('ZXZ', e1, degrees=False); R1 = r1.as_matrix()
    r2 = R.from_euler('ZXZ', e2, degrees=False); R2 = r2.as_matrix()
    return misorientation(R1, R2, out = out, symmetry_op = symmetry_op)


""" Steographic projection functions """
def misorientation(A, B, out='rad', symmetry_op:str='cubic'):
    """
    Original Author: Arthur Nishikawa (https://github.com/arthursn/pyebsd/blob/master/pyebsd/ebsd/orientation.py)
    Modified by: Emmanuel Atoleya Atindama (EAA)
    Calculates the misorientation between two rotation matrices, A & B
    Parameters
    ----------
    A : numpy ndarray shape(3, 3)
        First rotation matrix
    B : numpy ndarray shape(3, 3)
        Second rotation matrix
    out : str (optional)
        Unit of the output. Possible values are:
        'tr': as a trace value of the misorientation matrix
        'deg': as misorientation angle in degrees
        'rad': as misorientation angle in radians
        Default: 'deg'
    symmetry_op : str (optional) - Included by EAA
        Crystal symmefrom math import acos
import sys
import time
        symmetry of the material:
        'cubic': cubic symmetry has 24 symmetries
        'hexagonal': hexagonal symmetry has 12 symmetries
        Default: 'cubic'
    Returns
    -------
    misang : float
        Misorientation angle given in the unit specified by 'out'
    """
    if symmetry_op=='cubic':
        C = cubic_symmetry_operators()
    if symmetry_op=='hexagonal':
        C = hexagonal_symmetry_operators()
    if symmetry_op is None:
        C = no_symmetry_operators()
    Adim, Bdim = np.ndim(A), np.ndim(B)

    if (Adim == 2) and (Bdim == 2):
        # Tj = Cj * A * B^T
        T = np.tensordot(C, A.dot(B.T), axes=[[-1], [-2]])
        tr = T.trace(axis1=1, axis2=2)
        x = tr.max()  # Maximum trace
        # This might happen due to rounding error
        if x > 3.:
            x = 3.
        if out != 'tr':
            x = np.arccos((x-1.)/2.)  # mis x in radians
            if out == 'deg':
                x = np.degrees(x)  # mis x in degrees
    else:
        raise Exception('Invalid shapes of arrays A or B')
    return x


""" Symmetry operations for the hexagonal system """

def no_symmetry_operators():
    """
    Author: Emmanuel Atoleya Atindama (EAA)
    Identity matrix as symmetry operation, since we say no symmetry operations
    """
    axis_angle = [np.reshape([1,0,0,0,1,0,0,0,1], (3,3))]
    return np.stack(axis_angle, axis=0)


""" Symmetry operations for the cubic system """

def cubic_symmetry_operators():
    """
    Author: Emmanuel Atoleya Atindama (EAA)
    Lists symmetry matrices for cubic symmetry group in angle-axis format.
    A cubic crystal has 23 elements of symmetry, including 13 axes of symmetry. Include the identity, ans we have 24
    
    (1) Centre of symmetry : An imaginary point within the crystal such that any line drawn through it intersects the surface of the crystal at equal distances in both directions.
    (2) Plane of symmetry : It is an imaginary plane which passes through the centre of a crystal and divides it into two equal portions such that one part is exactly the mirror image of the other.
    A cubical crystal possesses six diagonal plane of symmetry and three rectangular plane of symmetry.  
    (3) Axis of symmetry : It is an imaginary straight line about which, if the crystal is rotated, it will present the same appearance more than once during the complete revolution. In general, if the same appearance of a crystal is repeated on rotating through an angle 360on, around an imaginary axis, the axis is called an n-fold axis.
    A cubical crystal possesses in all 13 axis of symmetry
    Axis of four-fold symmetry = 3 (Because of six faces)	Axis of three-fold symmetry = 4  (Because of eight corners)	Axis of two-fold symmetry = 6 (Because of twelve edges)
    (4) Elements of symmetry : The total number of planes, axes and centre of symmetry possessed by a crystal are termed as elements of symmetry. A cubic crystal possesses a total of 23 elements of symmetry.

    Planes of symmetry=(3+6)=9,
    Axes of symmetry=(3+4+6)=13,
    Centre of symmetry = 1.
    Total number of symmetry elements = 23
    """
    axis_angle = [# 3-fold axes only
                  np.reshape([1,0,0,0,1,0,0,0,1], (3,3)), np.reshape([0,0,1,1,0,0,0,1,0], (3,3)),
                   np.reshape([0,1,0,0,0,1,1,0,0], (3,3)), np.reshape([0,-1,0,0,0,1,-1,0,0], (3,3)),
                   np.reshape([0,-1,0,0,0,-1,1,0,0], (3,3)), np.reshape([0,1,0,0,0,-1,-1,0,0], (3,3)),
                   np.reshape([0,0,-1,1,0,0,0,-1,0], (3,3)), np.reshape([-1,0,0,0,0,1,0,1,0], (3,3)),
                   np.reshape([0,0,1,-1,0,0,0,-1,0],(3,3)),
                   # two-fold
                   np.reshape([-1,0,0,0,1,0,0,0,-1], (3,3)), np.reshape([-1,0,0,0,-1,0,0,0,1], (3,3)),
                   np.reshape([1,0,0,0,-1,0,0,0,-1], (3,3)), np.reshape([0,0,-1,0,-1,0,-1,0,0], (3,3)),
                   np.reshape([0,0,1,0,-1,0,1,0,0], (3,3)), np.reshape([0,0,1,0,1,0,-1,0,0], (3,3)),
                   np.reshape([0,0,-1,0,1,0,1,0,0], (3,3)), np.reshape([-1,0,0,0,0,-1,0,-1,0], (3,3)),
                   np.reshape([1,0,0,0,0,-1,0,1,0], (3,3)), np.reshape([1,0,0,0,0,1,0,-1,0], (3,3)),
                   np.reshape([-1,0,0,0,0,1,0,1,0], (3,3)), np.reshape([0,-1,0,-1,0,0,0,0,-1], (3,3)),
                   np.reshape([0,1,0,-1,0,0,0,0,1], (3,3)), np.reshape([0,1,0,1,0,0,0,0,-1], (3,3)),
                   np.reshape([0,-1,0,1,0,0,0,0,1], (3,3))
                  ]
    return np.stack(axis_angle, axis=0)
    

""" Symmetry operations for the hexagonal system """

def hexagonal_symmetry_operators():
    """
    Author: Emmanuel Atoleya Atindama (EAA)
    Lists symmetry matrices for hexagonal symmetry group in angle-axis format
    A hexagonal crystal has 11 elements of symmetry. Include the identity, and we have 12
    """
    a = np.sqrt(3)/2; b = 0.5
    axis_angle = [np.reshape([1,0,0,0,1,0,0,0,1], (3,3)), np.reshape([-b,a,0,-a,-b,0,0,0,1], (3,3)),
                  np.reshape([-b,-a,0,a,-b,0,0,0,1], (3,3)), np.reshape([b,a,0,-a,b,0,0,0,1], (3,3)),
                  np.reshape([-1,0,0,0,-1,0,0,0,1], (3,3)), np.reshape([b,-a,0,a,b,0,0,0,1], (3,3)),
                  np.reshape([-b,-a,0,-a,b,0,0,0,-1], (3,3)), np.reshape([1,0,0,0,-1,0,0,0,-1], (3,3)),
                  np.reshape([-b,a,0,a,b,0,0,0,-1], (3,3)), np.reshape([b,a,0,a,-b,0,0,0,-1], (3,3)),
                  np.reshape([-1,0,0,0,1,0,0,0,-1], (3,3)), np.reshape([b,-a,0,-a,-b,0,0,0,-1], (3,3))]
    return np.stack(axis_angle, axis=0)


def rotation_matrix_to_axis_angle(R):
    """
    Author: Emmanuel Atoleya Atindama (EAA)
    Convert a rotation matrix to axis-angle representation.
    Parameters:
    - R : 3x3 rotation matrix
    Returns:
    - axis : 1x3 array representing the axis of rotation
    - angle : angle of rotation in radians
    """
    # Calculate the trace of the rotation matrix
    trace = np.trace(R)
    # Calculate the angle of rotation
    angle = np.arccos((trace - 1) / 2.0)
    
    # Calculate the axis of rotation
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1]])
    axis /= (2 * np.sin(angle))
    return axis, angle


def axis_angle_to_rotation_matrix(axis, theta):
    """
    Original Author: Arthur Nishikawa (https://github.com/arthursn/pyebsd/blob/master/pyebsd/ebsd/orientation.py)
    Convert angle-axis representation to a rotation matrix.
    """
    theta_dim = np.ndim(theta)
    axis_dim = np.ndim(axis)

    if axis_dim != theta_dim + 1:
        raise Exception('Invalid shapes of theta or axis')

    if theta_dim == 0:
        theta = np.asarray(theta).reshape(-1)
        axis = np.asarray(axis).reshape(-1, 3)

    axis = axis/np.linalg.norm(axis, axis=1).reshape(-1, 1)

    N = len(theta); R = np.ndarray((N, 3, 3))

    ctheta = np.cos(theta)
    ctheta1 = 1 - ctheta
    stheta = np.sin(theta)

    R[:, 0, 0] = ctheta1*axis[:, 0]**2. + ctheta
    R[:, 0, 1] = ctheta1*axis[:, 0]*axis[:, 1] - axis[:, 2]*stheta
    R[:, 0, 2] = ctheta1*axis[:, 0]*axis[:, 2] + axis[:, 1]*stheta
    R[:, 1, 0] = ctheta1*axis[:, 1]*axis[:, 0] + axis[:, 2]*stheta
    R[:, 1, 1] = ctheta1*axis[:, 1]**2. + ctheta
    R[:, 1, 2] = ctheta1*axis[:, 1]*axis[:, 2] - axis[:, 0]*stheta
    R[:, 2, 0] = ctheta1*axis[:, 2]*axis[:, 0] - axis[:, 1]*stheta
    R[:, 2, 1] = ctheta1*axis[:, 2]*axis[:, 1] + axis[:, 0]*stheta
    R[:, 2, 2] = ctheta1*axis[:, 2]**2. + ctheta

    if theta_dim == 0: R = R.reshape(3, 3)
    return R
