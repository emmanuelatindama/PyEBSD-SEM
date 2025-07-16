# TODO docstring
# TODO functions to write to CTF

import numpy as np
from scipy.io import loadmat
from csv import reader as csv_reader
from datetime import date
from matplotlib.pyplot import savefig
#from pandas import read_csv
import os


def save_file(file):
    folder_name = str(date.today())
    old_path =os.getcwd()
    path = os.getcwd()+'/' + folder_name
    
    if os.path.exists(path)==False:
        folder = os.path.join(os.getcwd(), folder_name)
        os.makedirs(folder)
    
    os.chdir(path) 
    savefig(file+'.pdf')
    os.chdir(old_path)

# Case-sensitive. The Dream3D spec and .ctf files we've seen have all used the
# same case.
# TODO we don't really do anything about the fact that 2D is in degrees but 3D
# is in radians (in fact, this function doesn't care whether the angles are
# valid at all as long as they are real-valued floats). Should we handle that
# difference somewhere else? I think we should just always convert to and output
# radians.
# TODO the Dream3D .ctf specification doesn't say whether radians are in
# (0, 2*pi) or (-pi, pi). What to do?


def read_ctf(filename: str, missing_phase: int=None):
    """
    enter the name of the ctf file with extension '.ctf' to read the file
    """
    # TODO docstring
    file = open(filename)
    num_x          = None
    num_y          = None
    num_z          = None
    x_step         = None
    y_step         = None
    z_step         = None
    col_titles     = None
    n_header_lines = None
    for [line_number, line] in enumerate(file):
        split_line = line.split()
        if len(split_line) == 0: continue
        first_word = split_line[0]
        if   first_word == "XCells": num_x  = int(  split_line[1])
        elif first_word == "YCells": num_y  = int(  split_line[1])
        elif first_word == "ZCells": num_z  = int(  split_line[1])
        elif first_word == "XStep" : x_step = float(split_line[1])
        elif first_word == "YStep" : y_step = float(split_line[1])
        elif first_word == "ZStep" : z_step = float(split_line[1])
        # TODO what if CTF format spec doesn't require "Phase" to be in the
        # first column? Then this approach to detecting the data group header
        # can fail on perfectly valid files
        elif first_word == "Phase":
            col_titles = split_line
            # we expect this to be the last line in the header
            n_header_lines = line_number + 1 # line_number indexing is 0-based
            break
    file.close()
    print(num_x, num_y)
    if num_x  is None: raise ValueError("Failed to find XCells in header.")
    if num_y  is None: raise ValueError("Failed to find YCells in header.")
    if x_step is None: raise ValueError("Failed to find XStep in header.")
    if y_step is None: raise ValueError("Failed to find YStep in header.")
    volumetric = num_z is not None and num_z > 0
    if volumetric and z_step is None:
        raise ValueError("Failed to find ZStep in header.")
    if col_titles is None:
        raise ValueError("Failed to find data group header "
                         "(i.e. column titles).")
    try:
        x_col = col_titles.index("X")
        y_col = col_titles.index("Y")
        if volumetric: z_col = col_titles.index("Z")
        ang_cols = (col_titles.index("Euler1"),
                    col_titles.index("Euler2"),
                    col_titles.index("Euler3"))
        phase_col = 0 # TODO assumption. Does .ctf spec actually guarantee this?
    except ValueError as err:
        raise ValueError(f"Failed to find field {err.args[0].split()[0]} in "
                          "data group header (i.e. column titles).")

    data = np.loadtxt(filename, skiprows=n_header_lines)
    # TODO should the following 3 lines use `.nanmin()`? How do we even handle a
    # file containing nans?
    min_x = data[:, x_col].min()
    min_y = data[:, y_col].min()
    x_inds = ( (data[:, x_col] - min_x) / x_step ).astype(int)
    y_inds = ( (data[:, y_col] - min_y) / y_step ).astype(int)
    
    x_inds = np.array([i for j in range(num_y) for i in range(num_x)]); #print(x_inds[:50],'\n', x_inds_test[:50])
    y_inds = np.array([j for j in range(num_y) for i in range(num_x)]); #print(y_inds[:50],'\n', y_inds_test[:50])
    
    if volumetric:
        min_z = data[:, z_col].min()
        z_inds = ( (data[:, z_col] - min_z) / z_step ).astype(int)
        angles_shape = (num_x, num_y, num_z, 3)
        inds = (x_inds, y_inds, z_inds, slice(None))
    else:
        angles_shape = (num_x, num_y, 3)
        inds = (x_inds, y_inds, slice(None))
    # in case data doesn't exist for some indices
    angles = np.full(angles_shape, np.nan)
    angles[inds] = data[:, ang_cols]
    
    if missing_phase is None:
        # TODO is it safe to assume that any (0, 0, 0) Euler angles indicate a
        # missing value? It is theoretically possible for this to be known data
        missing_inds = np.all(angles == 0, axis=angles.ndim-1)
    else:
        missing_rows = data[:, phase_col] == missing_phase
        if volumetric:
            missing_inds = (x_inds[missing_rows],
                            y_inds[missing_rows],
                            z_inds[missing_rows],
                            slice(None))
        else:
            missing_inds = (x_inds[missing_rows],
                            y_inds[missing_rows],
                            slice(None))
    angles[missing_inds] = np.nan
    return angles




def read_mat(fname: str, filetype='none'):
    # TODO better docstring
    """Get the first variable in a .mat file
    input the file with the .mat extension
    ----------------------------------------------------
    All possibilities were not considered.
    If error message is returned, you may need to check code for bug
    """
    temp = loadmat(fname)
    try:
        return np.rad2deg(temp[filetype])
    except:
        return np.rad2deg(temp[list(temp.keys())[-1]])
        print('Error : check the key of the object holding the array and define it accordingly!')


def save_csv_as_ctf_file(csv_file, ctf_file=None, xstep=1, ystep=1, original_file=None):
    """
    Enter the name of the csv file with extension
    """
    csv_data = csv_reader( open( csv_file, 'rt' ) )
    
    num_x_cells = int(np.sqrt(len(list(csv_data))))
    num_y_cells = num_x_cells
    
    csv_data = csv_reader( open( csv_file, 'rt' ) )
    next(csv_data)
    
    phase  = np.empty( (num_x_cells, num_y_cells), dtype=int )
    angles = np.empty( (num_x_cells, num_y_cells, 3), dtype=float )
    
    for i in range(num_x_cells):
        for j in range(num_y_cells):
    # for j in range(num_y_cells):
    #     for i in range(num_x_cells):
            data_row = next(csv_data)
            phase[i,j] = int( data_row[0] )
            angles[i,j,0] = float( data_row[1] )
            angles[i,j,1] = float( data_row[2] )
            angles[i,j,2] = float( data_row[3] )
    
    
    if np.nanmax(angles) > 90:
        angles = np.deg2rad(angles)
    
    if ctf_file==None:
        ctf_file = csv_file[:-4]+'.ctf'
    
    save_ang_data_as_ctf( ctf_file, angles, xstep, ystep, phase, original_file=original_file )



    
def save_ang_data_as_ctf(filename, angles, xstep=1, ystep=1,
                  phase=None, mad=None, bc=None, bs=None, bands=None, error=None,
                  original_file=None, n_header_lines=15):
    """
    Enter the name of the ctf file you intend to give with the .ctf extention as a string the first variable,
    the second variable is the angle data array.
    filename: string of filename with the filepath
    """
    nx,ny = angles.shape[:2]
    int_zeros_array = np.zeros((nx,ny), dtype=int)

    if original_file is not None:
        f = open( original_file, 'rt' )
        header_lines = []
        for line in f:
            if line.split()[0] == 'Phase':
                header_lines.append(line)
                break
            header_lines.append(line)
        f.close()
        
        for line in header_lines:
            if line.startswith('XStep'):  xstep = float( line.split('\t')[1] )
            if line.startswith('YStep'):  ystep = float( line.split('\t')[1] )

    if phase is None:  phase = np.ones_like(int_zeros_array)
    if bands is None:  bands = int_zeros_array
    if error is None:  error = int_zeros_array
    if bc is None:  bc = int_zeros_array
    if bs is None:  bs = int_zeros_array
    if mad is None:  mad = np.ones((nx,ny), dtype=float)

    
    f = open(filename, 'wt')

    if original_file is not None:
        f.writelines( header_lines )
    else:
        f.write('Channel Text File\n')
        f.write('Prj\tC:\%s\n' % filename)
        f.write('Author	Unknown\n')
        f.write('JobMode\tGrid\n')
        f.write('XCells\t%d\nYCells\t%d\n' % (nx,ny))
        f.write('XStep\t%3.1f\nYStep\t%3.1f\n' % (xstep,ystep))
        f.write('AcqE1\t0\nAcqE2\t0\nAcqE3\t0\n')
        f.write('Euler angles refer to Sample Coordinate system (CS0)!\t'
                'Mag\t90\tCoverage\t100\tDevice\t0\t'
                'KV\t30\tTiltAngle\t70\tTiltAxis\t0\n')
        f.write('Phases	1\n')
        f.write('2.951;2.951;4.684\t90;90;120\tTitanium\t9\t194\t?\t'
                '??????.?.?	?????	Publication Info ???\n')
        f.write('Phase\tX\tY\tBands\tError\tEuler1\tEuler2\tEuler3\tMAD\tBC\tBS\n')

    angles = np.rad2deg(angles); angles[np.isnan(angles)] = 0
    for j in range(ny):
        for i in range(nx):
            x = i * xstep
            y = j * ystep
            euler1, euler2, euler3 = angles[i,j,:]
            if euler1 == 0 and euler2==0 and euler3 == 0:
                mad[i,j] = 0
                phase[i,j] = 0
                
            f.write("%d\t%s\t%s\t%d\t%d\t%s\t%s\t%s\t%s\t%d\t%d\n" %
                    (phase[i,j], str('%6.4f' % x)[:6], str('%6.4f' % y)[:6],
                     bands[i,j], error[i,j], str('%6.4f' % euler1)[:6],
                     str('%6.4f' % euler2)[:6], str('%6.4f' % euler3)[:6],
                     str('%6.4f' % mad[i,j])[:6], bc[i,j], bs[i,j] ))
            
    f.close()
    
    
    
    
def read_ang(filename, coord_index=(3,4), angle_index=(0,1,2),phase_index=None, bands_index=None,
	     error_index=None, mad_index=None, bc_index=None, bs_index=None, missing_index=None,
	     missing_label=0, skip_rows=0):
    """ 
    The code is for reading EBSD maps in .ang format
    The code is still in trial phase and so very unreliable.
    Use with caution.
    At the moment, only some parameters are active
    This version only handles 2D maps

    Parameters
    ----------
    filename : str - filename of the .ang file with the extension
    coord_index : tuple - column indices that tells the function where to find the coordinate index.
                  Should contain 2 values for a 2D EBSD map, and 3 values for a 3D map.
    angle_index : tuple - column indices that tells the function where to find the coordinate index.
                  Should contain 3 values for phi1, Phi, and phi2.
    missing_index: unsure of data type - index of missing values. Not currently in use.
                   Default is None.
    skip_rows : int - number of header rows present that need to be skipped
    
    Returns
    -------
    angles : ndarray - array containing Euler angles
    """
    
    data = np.loadtxt(filename, skiprows=skip_rows )

    x_ind, y_ind = coord_index
    angle_index = list(angle_index)

    x_coords = np.unique( data[:,x_ind] )
    y_coords = np.unique( data[:,y_ind] )
    min_x = x_coords.min()
    min_y = y_coords.min()
    num_x = len(x_coords)
    num_y = len(y_coords)
    x_step = x_coords[1] - x_coords[0]
    y_step = y_coords[1] - y_coords[0]
    
    # angles = np.zeros( (nx,ny,3) )
    #===========================================================================
    print(num_x, num_y)
    if num_x  is None: raise ValueError("Failed to find XCells in header.")
    if num_y  is None: raise ValueError("Failed to find YCells in header.")
    if x_step is None: raise ValueError("Failed to find XStep in header.")
    if y_step is None: raise ValueError("Failed to find YStep in header.")
    
    
    min_x = data[:, x_ind].min()
    min_y = data[:, y_ind].min()
    x_inds = ( (data[:, x_ind] - min_x) / x_step ).astype(int)
    y_inds = ( (data[:, y_ind] - min_y) / y_step ).astype(int)
    
    x_inds = np.array([i for j in range(num_y) for i in range(num_x)]); #print(x_inds[:50],'\n', x_inds_test[:50])
    y_inds = np.array([j for j in range(num_y) for i in range(num_x)]); #print(y_inds[:50],'\n', y_inds_test[:50])
    
    angles_shape = (num_x, num_y, 3)
    inds = (x_inds, y_inds, slice(None))

    # in case data doesn't exist for some indices
    angles = np.full(angles_shape, np.nan)
    angles[inds] = data[:, angle_index]
    
    if missing_index is None:
        # TODO is it safe to assume that any (0, 0, 0) Euler angles indicate a
        # missing value? It is theoretically possible for this to be known data
        missing_inds = np.all(angles == 0, axis=angles.ndim-1)
    
    angles[missing_inds] = np.nan
    return angles

    
