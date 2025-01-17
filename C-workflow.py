# End-To-End Modular 3D Visualisation Workflow for Simulated, 
# Fixed-Directional and Hybrid-Rotational GPR Data Processing 
import os 
import glob
import shutil
import h5py
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tkinter as tk
from datetime import datetime
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from scipy.constants import speed_of_light as c0
from scipy.spatial import ConvexHull
from sklearn.cluster import HDBSCAN
from sklearn.datasets import make_blobs

# ---------------------------------------------------------
# SETUP | HOUSEKEEPING
# ---------------------------------------------------------
# gprmax --> simulated GPR data
# zond ----> fixed-directional GPR data
# vna -----> hybrid-rotational GPR data 

# /// Use Same HOUSEKEEPING as presented in Appendix A ///

# ---------------------------------------------------------
# EXTRACTION AND ALIGNMENT
# ---------------------------------------------------------
# local coords - relative to antenna starting location
# global coords - relative to corner marker datum on Region of 
# Interest (RoI) 

# Functions to Trigger Position Data Recovery 
def xyz_gprmax(dirPath):
    """
    Triggers recovery of position data.

    Parameters: 
    dirPath (string): Current working directory filepath

    Returns:
    xyz (array): Cartesian coords. 
    lT (array): Spatial-temporal disp.s

    """  

    update('recovering gprmax xyz coords...')

    # -------------------

    # Checkpointing (if possible)
    try:
        filePath = os.path.join(dirPath,'input.h5')
        xyz = h5In(filePath,'xyz')
        lT = h5In(filePath,'lT')
    except:
        
        # list files 
        filePaths, _ = getFiles(
            os.path.join(dirPath,'transects'),['.out'])
        nFiles = len(filePaths)

        # -------------------

        # initialise storage
        xyz = []; lT = []

        # recover data
        if (nFiles > 0):
            xyz, lT = getXYZ_gprmax(dirPath, filePaths)

        # -------------------

        # export
        filePath = os.path.join(dirPath,'input.h5')
        h5Out(filePath,xyz,'xyz')
        h5Out(filePath,lT,'lT')

    # -------------------
    return xyz, lT
def xyz_vna(dirPath):
    """
    Triggers recovery of position data.

    Parameters: 
    dirPath (string): Current working directory filepath

    Returns:
    xyz (array): Cartesian coords. 
    lT (array): Spatial-temporal disp.s
    rte (array): Cylindrical coords.

    Notes:

    coords. 
        (-) <l> is sample disp. [m] from vna port 1
        (-) <T> is sample temporal disp. [s] from vna 
                port 1
        (-) <r> is sample radial disp. [m] from centre of 
                rotation (i.e. antenna base)

    define offset (local --> global)
        (-) <xo> assumes scan setup s.t. antenna boresight begins 
            at RoI datum (i.e. the global datum) in x direction 
        (-) <yo> (vertical) assumes sample points with well-defined 
            coordinates (i.e. r>0) begin form centre of rotation
        (-) <z0> assumes antenna boresight is aligned to centre of 
            RoI in z direction

    """   

    update('recovering vna/rotational xyz coords...')

    # -------------------

    # Checkpointing (if possible)
    try:
        filePath = os.path.join(dirPath,'input.h5')
        xyz = h5In(filePath,'xyz')
        rte = h5In(filePath,'rte')
        lT = h5In(filePath,'lT')
    except:

        # list files
        filePaths, _ = getFiles(
            os.path.join(dirPath,'traces'),['.s1p'])
        nFiles = len(filePaths)
        
        # -------------------

        # initialise storage
        xyz = []; lT = []; rte = []

        # recover data
        if (nFiles > 0):
            xyz, lT, rte = getXYZ_vna(dirPath, filePaths)

        # -------------------

        # export
        filePath = os.path.join(dirPath,'input.h5')
        h5Out(filePath,xyz,'xyz')
        h5Out(filePath,rte,'rte')
        h5Out(filePath,lT,'lT')

    # -------------------
    return xyz, rte, lT
def xyz_zond(dirPath):
    """
    Triggers recovery of position data.

    Parameters: 
    dirPath (string): Current working directory filepath

    Returns:
    xyz (array): Cartesian coords. 
    lT (array): Spatial-temporal disp.s

    """   

    update('recovering zond-GPR xyz coords...')
    
    # -------------------
    
    # Checkpointing (if possible)
    try:
        filePath = os.path.join(dirPath,'input.h5')
        xyz = h5In(filePath,'xyz')
        lT = h5In(filePath,'lT')
    except:

        # list files
        filePaths, _ = getFiles(
            os.path.join(dirPath,'transects'),['.sgy'])
        nFiles = len(filePaths)

        # -------------------

        # initialise storage
        xyz = []; lT = []

        # recover data
        if (nFiles > 0):
            xyz, lT = getXYZ_zond(dirPath, filePaths)

        # -------------------

        # export
        filePath = os.path.join(dirPath,'input.h5')
        h5Out(filePath,xyz,'xyz')
        h5Out(filePath,lT,'lT')

    # -------------------
    return xyz, lT

# Functions to Trigger Amplitude Data Recovery
def A_gprmax(dirPath):
    """
    Triggers recovery of amplitude data.

    Parameters: 
    dirPath (string): Current working directory filepath

    Returns:
    A1 (array): Amplitude data -  main scanning pass

    """   

    update('recovering gprmax amplitudes...')
    
    # ------------------

    # Checkpointing (if possible)
    try:
        filePath = os.path.join(dirPath,'input.h5')
        A1 = h5In(filePath,'A1')
    except:

        # list files 
        filePaths, _ = getFiles(
            os.path.join(dirPath,'transects'),['.out'])
        nFiles = len(filePaths)

        # ------------------

        # initialise storage
        A1 = []

        # recover data
        if (nFiles > 0):
            A1 = getA_gprmax(filePaths)
            
        # -------------------

        # export 
        filePath = os.path.join(dirPath,'input.h5')
        h5Out(filePath,A1,'A1')

    # ------------------
    return A1
def A_vna(dirPath):
    """
    Triggers recovery of amplitude data.

    Parameters: 
    dirPath (string): Current working directory filepath

    Returns:
    A0 (array): Amplitude data - background calibration scan
    A1 (array): Amplitude data -  main scanning pass

    Notes:

    subtract backgrounds from measurements
        (-) suppresses prevalence of noise associated with 
            subsurface background media
        (-) suppresses prevalence of noise associated with 
            measurement cable flex

    """    

    update('recovering vna/rotational amplitudes...')
    
    # -------------------

    # Checkpointing (if possible)
    try:
        filePath = os.path.join(dirPath,'input.h5')
        A0 = h5In(filePath,'A0')
        A1 = h5In(filePath,'A1')
    except:

        # list files - background(s)
        filePaths_0, _ = getFiles(
            os.path.join(dirPath,'traces_background'),['.s1p'])
        nFiles_0 = len(filePaths_0)

        # list files - measurement(s)
        filePaths_1, _ = getFiles(
            os.path.join(dirPath,'traces'),['.s1p'])
        nFiles_1 = len(filePaths_1)

        # ------------------- 

        # initialise amplitude storage
        A0 = []
        A1 = []

        # recover t-domain amplitudes - background(s)
        # note: complex-valued
        if (nFiles_0 > 0):
            A0 = getA_vna(filePaths_0)    

        # recover t-domain amplitudes - measurements(s)
        # note: complex-valued
        if (nFiles_1 > 0):
            A1 = getA_vna(filePaths_1)

        # -------------------

        # export 
        filePath = os.path.join(dirPath,'input.h5')
        h5Out(filePath,A0,'A0')
        h5Out(filePath,A1,'A1')

    # -------------------
    return A0, A1
def A_zond(dirPath):
    """
    Triggers recovery of amplitude data.

    Parameters: 
    dirPath (string): Current working directory filepath

    Returns:
    A1 (array): Amplitude data - main scanning pass

    """ 

    update('recovering zond-GPR amplitudes...')
    
    # ------------------
    
    # Checkpointing (if possible)
    try:
        filePath = os.path.join(dirPath,'input.h5')
        A1 = h5In(filePath,'A1')
    except:

        # list files
        filePaths, _ = getFiles(
            os.path.join(dirPath,'transects'),['.sgy'])
        nFiles= len(filePaths)
        
        # ------------------

        # initialise storage
        A1 = []

        # recover data
        if (nFiles > 0):
            A1 = getA_zond(filePaths) 

        # -------------------

        # export
        filePath = os.path.join(dirPath,'input.h5')
        h5Out(filePath,A1,'A1')

    # ------------------
    return A1

# Functions to Recover Sample/Trace/Transect Indexes
def iSAB_gprmax(dirPath):
    """
    Returns scan sample/trace/transect indexes.

    Parameters: 
    dirPath (string): Current working directory filepath

    Returns:
    iSAB (array): sample/trace/transect indexes 

    """  

    update('recovering iSAB indexes...')
    
    # -------------------

    # Checkpointing (if possible)
    try:
        filePath = os.path.join(dirPath,'input.h5')
        iSAB = h5In(filePath,'iSAB')
    except: 
        
        # list files 
        filePaths, _ = getFiles(
            os.path.join(dirPath,'transects'),['.out'])

        # -------------------

        # recover global parameters
        nS_A, _ = getGlobals_gprmax(filePaths[0])
        nB = len(filePaths)

        # recover local variables 
        nA_B, nA = getLocals_gprmax(filePaths)
        nS_B = nS_A*nA_B                                    
        nS = nS_A*nA    

        # pre-allocate storage array
        iSAB = np.zeros((nS,3),dtype=np.int16) 

        # -------------------

        # compute indexes
        iSAB[:,0] = np.tile(np.arange(0,nS_A),nA)  
        iSAB[:,1] = np.tile(np.repeat(np.arange(0,nA_B),nS_A),nB) 
        iSAB[:,2] = np.repeat(np.arange(0,nB),nS_B)    

        # -------------------

        # export
        filePath = os.path.join(dirPath,'input.h5')
        h5Out(filePath,iSAB,'iSAB')            

    # -------------------
    return iSAB
def iSAB_vna(dirPath):
    """
    Returns scan sample/trace/transect indexes.

    Parameters: 
    dirPath (string): Current working directory filepath

    Returns:
    iSAB (array): sample/trace/transect indexes 

    """ 

    update('recovering iSAB indexes...')
    
    # -------------------

    # Checkpointing (if possible)
    try:
        filePath = os.path.join(dirPath,'input.h5')
        iSAB = h5In(filePath,'iSAB')
    except: 

        # list files
        filePaths, _ = getFiles(
            os.path.join(dirPath,'traces'),['.s1p'])

        # -------------------

        # recover global parameters
        nS_A, _, _, _ = getGlobals_vna(filePaths[0])
        nA = len(filePaths)

        # recover local variables
        nA_B, nB = getLocals_vna(filePaths)
        nS_B = nS_A*nA_B
        nS = nS_B*nB

        # pre-allocate storage
        iSAB = np.zeros((nS,3),dtype=np.int16)

        # -------------------

        # compute indexes
        iSAB[:,0] = np.tile(np.arange(0,nS_A),nA)
        iSAB[:,1] = np.tile(np.repeat(np.arange(0,nA_B),nS_A),nB)
        iSAB[:,2] = np.repeat(np.arange(0,nB),nS_B)

        # -------------------

        # export 
        filePath = os.path.join(dirPath,'input.h5')
        h5Out(filePath,iSAB,'iSAB')      

    # -------------------
    return iSAB
def iSAB_zond(dirPath):
    """
    Returns scan sample/trace/transect indexes.

    Parameters: 
    dirPath (string): Current working directory filepath

    Returns:
    iSAB (array): sample/trace/transect indexes 

    Notes:

    if exporting <iSAB> to .h5 file, elements may not appear in 
    HDF5 viewer
        (-) suspect too many to display on PC
        (-) if file size is large - approx. O(MB) - data is present
        (-) data can be recovered and viewed by importing .h5 
            file to python  
        (-) suspect HDF5 viewer application has limit on no. 
            elements it can render in one dataset
        (-) suspect = defaults to show no elements if rendering 
            limit exceeded

    """ 

    update('recovering iSAB indexes...')
    
    # -------------------

    # Checkpointing (if possible)
    try:
        filePath = os.path.join(dirPath,'input.h5')
        iSAB = h5In(filePath,'iSAB')
    except: 

        # list files
        filePaths, _ = getFiles(
            os.path.join(dirPath,'transects'),['.sgy'])

        # -------------------

        # recover global parameters
        _, _, _, _, nS_A, _, _, _ = getGlobals_zond(filePaths[0])
        nB = len(filePaths)

        # recover local variables
        nA_B, nA = getLocals_zond(filePaths)                                                                    
        nS = nS_A*nA                                                
        
        # pre-allocate storage array
        iSAB = np.zeros((nS,3),dtype=np.int16)                                     

        # -------------------

        # compute indexes - samples
        iSAB[:,0] = np.tile(np.arange(0,nS_A),nA)

        # -------------------
        
        # compute indexes - traces and transects
        # recall: <nA_B> an array, transects are different lengths
        for i in range(nB):

            # indexes of samples in ith transect
            i0 = (nS_A)*np.sum(nA_B[0:i])
            i1 = i0 + (nS_A*nA_B[i])
            nS_Bi = i1 - i0

            # populate trace count
            iSAB[i0:i1,1] = np.repeat(np.arange(0,nA_B[i]),nS_A)

            # populate transect count
            iSAB[i0:i1,2] = np.repeat(i,nS_Bi)

        # -------------------

        # export
        filePath = os.path.join(dirPath,'input.h5')
        h5Out(filePath,iSAB,'iSAB')       

    # -------------------
    return iSAB

# Functions to Recover Linked Parameters | Global
def getGlobals_gprmax(filePath):
    """
    Returns scan global linked parameters. 

    Parameters: 
    filePath (string): Specific input file
    
    Returns:
    nS_A (integer): No. samples per trace [1] 
    dt (float): Temporal incrementation [s]

    """ 

    # Extract hdf5 file attributes
    with h5py.File(filePath,'r') as h5:
        nS_A = h5.attrs['Iterations']
        dt = h5.attrs['dt']

    # -------------------
    return nS_A, dt
def getGlobals_vna(filePath):
    """
    Returns scan global linked parameters. 

    Parameters: 
    filePath (string): Specific input file
    
    Returns:
    nS_A (integer): No. samples per trace [1]
    dt (float): Temporal incrementation [s]
    fmt (string): data format handle 
    f (array): frequency domain values [Hz]

    """ 

    # Define header block
    headerRows = 12

    # Import trace data
    with open(filePath,'r') as file:
        header = [file.readline().strip() \
                  for _ in range(headerRows)]
        data = np.loadtxt(file)

    # ------------------- 

    # Extract amplitude data format
    fmt = header[11][7:9]
    del header

    # Extract frequency values
    f = data[:,0]
    del data

    # Recover no. samples per trace
    nS_A = len(f)

    # Recover temporal inc. between successive samples
    df = f[1]-f[0]
    T = 1/df
    dt = T/(nS_A-1)  
    del df, T  

    # -------------------
    return nS_A, dt, fmt, f
def getGlobals_zond(filePath):
    """
    Returns scan global linked parameters. 

    Parameters: 
    filePath (string): Specific input file
    
    Returns:
    nb_H0 (integer): No. bytes in global header (EBCDIC block) [1] 
    nb_H1 (integer): No. bytes in global header [1]
    nb_H (integer): Total no. bytes in global header block [1] 
    nb_h (integer): No. bytes in trace header block [1]
    nS_A (integer): No. samples per trace [1]
    nb_S (integer): No. bytes per sample [1]
    dt (float): Temporal incrementation [s]
    dA (float): Lateral disp. incrementation [m]

    Notes:

    disp. scalar <s> for odometer disp. <l>
        (-) if s > 0 --> l = s*l
        (-) if s < 0 --> l = l/abs(s)  

    disp. inc. between traces <dA>
        (-) read off the disp.  at the first trace recorded
        (-) note zond-GPR does not record an initial trace at 
            0m disp., first trace at disp. <dA>
        (-) assumes the same trace disp. across all transects 

    """ 

    # Define expected byte-block sizes
    nb_H0 = 3200                            # EBCDIC header 
    nb_H1 = 400                             # service info header
    nb_H = nb_H0 + nb_H1                    # global header             
    nb_h = 240                              # trace header(s)

    # -------------------

    # Recover no. samples per trace [1]
    # note: this is needed for sizing trace block size
    # note: <int> applied to handle arbitrary precision
    nS_A = readBytes(filePath,3220,'int16',2)
    nS_A = int(nS_A)                                   

    # Recover no. bytes per sample (for trace amplitude data) [1]
    # note: this is needed for sizing trace block size
    dataFormatCode  = readBytes(filePath,3224,'int16',2)
    dataFormatCode = int(dataFormatCode)
    if (dataFormatCode == 3):
        nb_S = 2
    elif (dataFormatCode == 2):
        nb_S = 4
    else:
        raise Exception(f'unsupported <dataFormatCode>: \
                        {dataFormatCode}')
    del dataFormatCode

    # Recover temporal inc. between successive samples [s]
    # note: binary data value stored in no. picoseconds
    dt = readBytes(filePath,3216,'int16',2)
    dt = dt*(1e-12)

    # Recover spatial inc. between successive traces [m]
    # note: disp. scalar <s> recovers <dA> in [m]
    s = readBytes(filePath,(nb_H + 70),'int16',2)              
    dA = readBytes(filePath,(nb_H + 72),'int32',4)           
    if (s > 0):
        dA = s*dA
    elif (s < 0):
        dA = dA/(np.abs(s))
    else:
        raise Exception(f'unsupported disp. scalar: {s}')
    del s

    # -------------------
    return nb_H0, nb_H1, nb_H, nb_h, nS_A, nb_S, dt, dA

# Functions to Recover Linked Parameters | Local
def getLocals_gprmax(filePaths):
    """
    Returns scan local linked parameters. 

    Parameters: 
    filePaths (array): List of relevant input files
    
    Returns:
    nA_B (integer): No. traces per transect [1]
    nA (integer): No. traces total in full scanning pass [1]

    """ 

    # Recover no. transects
    nB = len(filePaths)

    # -------------------

    # Import transect amplitude array
    with h5py.File(filePaths[0],'r') as h5:
        data = h5['rxs']['rx1']['Ez'][0,:]
    
    # Read off no. traces per transect
    # note: always same for all traces in gprMax grid scan
    nA_B = np.shape(data)[0]

    # -------------------

    # Compute total no. traces across all transects
    nA = nA_B*nB

    # -------------------
    return nA_B, nA
def getLocals_vna(filePaths):
    """
    Returns scan local linked parameters. 

    Parameters: 
    filePaths (array): List of relevant input files
    
    Returns:
    nA_B (integer): No. traces per transect [1]
    nB (integer): No. transects total in full scanning pass [1]

    """   

    # Recover filename of final trace
    fileName = os.path.basename(filePaths[-1])

    # Recover trace and transect index values
    parts = fileName.split('_')
    part_b, part_a = parts[1], parts[2]
    part_a = part_a.split('.')[0]
    b = int(part_b[1:])
    a = int(part_a[1:])   
    del parts, part_b, part_a

    # Recover no. traces per transect
    # recall: indexing starts from 0
    nA_B = a + 1
     
    # Recover total no. transects
    # recall: indexing starts from 0
    nB = b + 1

    # -------------------
    return nA_B, nB
def getLocals_zond(filePaths):
    """
    Returns scan local linked parameters. 

    Parameters: 
    filePaths (array): List of relevant input files
    
    Returns:
    nA_B (array): No. traces per transect
    nA (integer): No. traces total in full scanning pass [1]

    Notes: 

    Variability
        (-) nA_B is variable transect by transect in this scenario. 

    """      

    # Recover global parameters
    _, _, nb_H, nb_h, nS_A, nb_S, _, _ = \
        getGlobals_zond(filePaths[0])

    # Recover total no. transects 
    nB = len(filePaths) 

    # Pre-allocate storage
    nA_B = np.zeros(nB,dtype='int16')

    # -------------------

    # Cycle transects...
    for i in range(nB):

        # compute no. traces from filesize
        nb_F = os.path.getsize(filePaths[i])               
        nA_i = (nb_F-nb_H)/(nb_h+(nS_A*nb_S))   

        # store value
        nA_B[i] = nA_i

    # -------------------

    # Compute total no. traces across all transects 
    nA = np.sum(nA_B)                                         
    
    # -------------------
    return nA_B, nA

# Functions to Recover Scan Metadata
def getMeta_gprmax(dirPath):
    """
    Returns scan metadata. 

    Parameters: 
    dirPath (string): Current working directory filepath
    
    Returns:
    M (array): Metadata parameters

    """   

    # Locate metadata file
    filePath = os.path.join(dirPath,'metadata','parameters.txt')

    # Import metadata
    M = np.loadtxt(filePath)

    # ------------------

    # Check for valid antenna separation
    if (M[5] <= 0):
        raise Exception(f'invalid antenna separation: {M[5]}')

    # ------------------

    # Metadata breakdown - summary
    """
    M[00]... <nB0>........ | no. transect rows in grid scan [1]
    M[01]... <nB1>........ | no. transect cols in grid scan [1]
    M[02]... <dB0>........ | grid rows separation [m]
    M[03]... <dB1>........ | grid cols separation [m]
    M[04]... <epsilon_r>.. | relative electrical permittivity [1]
    M[05]... <G>.......... |  distance between tx and rx [m]
    """

    # Metadata breakdown - syntax
    """
    nB0 = M[0]
    nB1 = M[1]
    dB0 = M[2]
    dB1 = M[3]
    epsilon_r = M[4]
    G = M[5]
    """

    # ------------------ 
    return M
def getMeta_vna(dirPath):
    """
    Returns scan metadata. 

    Parameters: 
    dirPath (string): Current working directory filepath
    
    Returns:
    M (array): Metadata parameters

    """  

    # Locate metadata file
    filePath = os.path.join(dirPath,'metadata','parameters.txt')

    # Import metadata
    M = np.loadtxt(filePath)

    # ------------------

    # Check for valid scanType
    if (M[6] != 0) and (M[6] != 1):
        raise Exception(f'invalid scanType: {M[6]}')

    # ------------------

    # Metadata breakdown - summary
    """
    M[00]... <x0/e0> | initial lateral disp. from home [m]
    M[01]... <x1/e1> | final  lateral disp. from home [m]
    M[02]... <t0>... | initial azimuthal disp. from home [deg]
    M[03]... <t1>... | final azimuthal disp. from home [deg]
    M[04]... <dx/de> | lateral advance spatial inc. [m]
    M[05]... <dt>... | azimuthal advance spatial inc.  [deg] 
    M[06]... <scanType>... | flag: 0 = rotational, 1 = planar
    M[07]... <epsilon_r>.. | relative electrical permittivity [1]
    M[08]... <l_antenna>.. | length of antenna body [m]
    M[09]... <l_down>..... | distance aperture to surface [m]
    M[10]... <t_down>..... | azimuth at downward vertical [deg]
    M[11]... <l_RoIx>..... | x span of sample region of interest [m]
    M[12]... <l_RoIy>..... | y span of sample region of interest [m] 
    M[13]... <l_RoIz>..... | z span of sample region of interest [m]
    """

    # Metadata breakdown - syntax
    """
    x0 = M[0]
    x1 = M[1]
    t0 = M[2]
    t1 = M[3]
    dx = M[4]
    dt = M[5]
    scanType = M[6]
    epsilon_r = M[7]
    l_antenna = M[8]                                           
    l_down = M[9]                                              
    t_down = M[10]                                  
    l_RoIx = M[11]
    l_RoIy = M[12]
    l_RoIz = M[13]
    """
    
    # ------------------ 
    return M
def getMeta_zond(dirPath):
    """
    Returns scan metadata. 

    Parameters: 
    dirPath (string): Current working directory filepath
    
    Returns:
    M (array): Metadata parameters

    """  

    # Locate metadata file
    filePath = os.path.join(dirPath,'metadata','parameters.txt')

    # Import metadata
    M = np.loadtxt(filePath)

    # ------------------

    # Check distance between midpoint and wheel well-defined
    if (M[5] <= 0):
        raise Exception(f'invalid antenna midpoint-wheel \
                        separation: {M[5]}')

    # Check antenna body length well-defined
    if (M[6] <= 0):
        raise Exception(f'invalid antenna long side length: {M[6]}')

    # ------------------

    # Metadata breakdown - summary
    """
    M[00]... <nB0>........ | no. transect rows in grid scan [1]
    M[01]... <nB1>........ | no. transect cols in grid scan [1]
    M[02]... <dB0>........ | grid rows separation [m]
    M[03]... <dB1>........ | grid cols separation [m]
    M[04]... <epsilon_r>.. | relative electrical permittivity of 
                                subsurface [1] selected in Prism2
    M[05]... <G>.......... | absolute distance antenna midpoint and 
                                odometer wheel contact point [m]
    M[06]... <L>.......... | antenna body longest side length [m]
    M[07]... <Lx>......... | RoI length in x [m] (from x = 0)
    M[07]... <Ly>......... | RoI depth in y [m] (from y = 0)
    M[07]... <Lz>......... | RoI length in z [m] (from z = 0)
    """

    # Metadata breakdown - syntax
    """
    nB0 = M[0]
    nB1 = M[1]
    dB0 = M[2]
    dB1 = M[3]
    epsilon_r = M[4]
    G = M[5]
    L = M[6]
    """

    # ------------------
    return M

# Functions to DIRECTLY Recover Position Data 
def getXYZ_gprmax(dirPath,filePaths):
    """
    Returns position data directly from input files.  

    Parameters: 
    dirPath (string): Current working directory filepath
    filePaths (array): List of relevant input files
    
    Returns:
    xyz (array): Cartesian coords. 
    lT (array): Spatial-temporal disp.s


    """  

    # Import scan metadata
    M = getMeta_gprmax(dirPath)

    # Unpack key metadata
    nB0 = int(M[0])
    nB1 = int(M[1])
    epsilon_r = M[4]
    G = M[5]                 

    # Clean-up
    del M

    # ------------------

    # Recover global parameters
    nS_A, dt = getGlobals_gprmax(filePaths[0])

    # Recover local variables 
    nA_B, nA = getLocals_gprmax(filePaths)
    nS = nS_A*nA                                    

    # Pre-allocate storage array
    xyz = np.zeros((nS,3))   
    lT = np.zeros((nS,2))

    # ------------------

    # Compute temporal disp. along trace
    T = np.linspace(0,dt*(nS_A-1),nS_A)  

    # Compute spatial disp. along trace 
    v = c0/(np.sqrt(epsilon_r))                                                             
    l = (v*T)/2                                              

    # Extend across all traces
    T = np.tile(T,nA)
    l = np.tile(l,nA)

    # Populate y coords. 
    # recall: downward is -ve                                        
    xyz[:,1] = -l

    # Populate spatial/temporal disp. coords. 
    lT[:] = np.column_stack((l,T))
    
    # Clean-up
    del T, v, l

    # -------------------

    # List trace files 
    filePaths, _ = getFiles(
        os.path.join(dirPath,'traces'),['.out'])

    # -------------------

    # Cycle traces...
    for i in range(nA):

        # extract rx antenna coordinates
        with h5py.File(filePaths[i],'r') as h5:
            xz = h5['rxs']['rx1'].attrs['Position'][[0,2]]

        # populate storage
        i0 = 0 + i*nS_A
        i1 = i0 + nS_A
        xyz[i0:i1,[0,2]] = np.tile(xz,(nS_A,1))

        # progress update
        progress(nA,i,'...')
    
    # Cleanup 
    del filePaths

    # -------------------

    # Compute no. samples across transect rows/cols
    nS_B0 = nS_A*nA_B*nB0
    nS_B1 = nS_A*nA_B*nB1

    # Define antenna separation vector
    # note: changes when rows --> cols
    g0 = np.array([G,0,0])
    g0 = np.tile(g0,(nS_B0,1))
    g1 = np.array([0,0,G])
    g1 = np.tile(g1,(nS_B1,1))
    g = np.concatenate((g0,g1),axis=0)
    del g0, g1 

    # Offset xz-coords. using antenna separation vector
    # note: maps coords. from rx antenna to rx/tx midpoint
    # note: rx antenna is in front of tx antenna by convention
    xyz = xyz - g/2

    # -------------------
    return xyz, lT
def getXYZ_vna(dirPath,filePaths):
    """
    Returns position data directly from input files.  

    Parameters: 
    dirPath (string): Current working directory filepath
    filePaths (array): List of relevant input files
    
    Returns:
    xyz (array): Cartesian coords. 
    lT (array): Spatial-temporal disp.s
    rte (array): Cylindrical coords. 

    """  

    # Import scan metadata
    M = getMeta_vna(dirPath)

    # Unpack key metadata
    x0 = M[0]
    x1 = M[1]
    t0 = M[2]
    t1 = M[3]
    scanType = M[6]
    l_antenna = M[8]                                           
    l_down = M[9]                                              
    t_down = M[10]                                  
    l_RoIz = M[13]

    # Clean-up
    del M

    # -------------------

    # Recover global parameters
    nS_A, _ , _, _ = getGlobals_vna(filePaths[0])
    nA = len(filePaths)

    # Recover local variables
    nA_B, nB = getLocals_vna(filePaths)
    nS = nS_A*nA
    nS_B = nS_A*nA_B 
    
    # Pre-allocate storage
    xyz = np.zeros((nS,3))
    lT = np.zeros((nS,2))
    rte = np.zeros((nS,3))

    # -------------------

    # Compute local coords. 
    if (scanType == 0):

        # lateral values 
        e = np.repeat(np.linspace(x0,x1,nB),nS_B) 

        # azimuthal values - relative to azimuthal home
        t = np.linspace(t0,t1,nA_B)

        # azimuthal values - relative to vertical downward
        # note: follows argand diagram, azimuths after downward 
        # vertical are negative
        t =  t_down - t

        # azimuthal values - radians
        t = np.deg2rad(t)

        # azimuthal values 
        t = np.tile(np.repeat(t,nS_A),nB)                                         
    else:
        
        # lateral values
        e = np.tile(np.repeat(np.linspace(x0,x1,nA_B),nS_A),nB) 

        # azimuthal values - relative to azimuthal home
        t = np.linspace(t0,t1,nB)

        # azimuthal values - relative to vertical downward
        # note: follows argand diagram, azimuths after downward 
        # vertical are negative
        t = t - t_down

        # azimuthal values - radians
        t = np.deg2rad(t)

        # azimuthal values 
        t = np.repeat(t,nS_B) 
    T,l,r = getTlr_vna(dirPath)
    lT[:] = np.column_stack((l,T))
    rte[:] = np.column_stack((r,t,e)) 

    # Clean-up
    del T, l

    # -------------------

    # Convert local coords. to cartesian
    # recall: downward is -ve, azimuth relative to downward vertical
    # note: valid provided within (-90,90) of downward vertical
    x = e
    y = -(r)*np.cos(t)
    z = r*np.sin(t)
    xyz = np.column_stack((x,y,z)) 

    # Clean-up
    del r,t,e,x,y,z

    # -------------------

    # Define offset (local --> global)
    ox = -x0                                    
    oy = l_down + l_antenna                     
    oz = l_RoIz/2                              

    # Compute global coords.
    xyz = xyz + np.array([ox,oy,oz])

    # -------------------
    return xyz, lT, rte
def getXYZ_zond(dirPath,filePaths):
    """
    Returns position data directly from input files.  

    Parameters: 
    dirPath (string): Current working directory filepath
    filePaths (array): List of relevant input files
    
    Returns:
    xyz (array): Cartesian coords. 
    lT (array): Spatial-temporal disp.s


    """  
     
    # Import scan metadata
    M = getMeta_gprmax(dirPath)

    # Unpack key metadata
    nB0 = int(M[0])
    nB1 = int(M[1])
    dB0 = M[2]
    dB1 = M[3]
    epsilon_r = M[4]
    G = M[5]
    L = M[6]

    # Clean-u 
    del M              

    # -------------------

    # Recover global parameters
    _, _, _, _, nS_A, _, dt, dA = getGlobals_zond(filePaths[0])
    nB = nB0 + nB1

    # Recover local variables
    nA_B, nA = getLocals_zond(filePaths)                       
    nS = nS_A*nA                                               

    # Pre-allocate storage array
    xyz = np.zeros((nS,3))
    lT = np.zeros((nS,2))                                     

    # -------------------

    # Compute temporal disp. along trace
    T = np.linspace(0,dt*(nS_A-1),nS_A) 

    # Compute spatial disp. along trace 
    v = c0/(np.sqrt(epsilon_r))                                                             
    l = (v*T)/2

    # Extend across all traces
    T = np.tile(T,nA)
    l = np.tile(l,nA)

    # Populate y coords.
    # recall: downward is -ve                                        
    xyz[:,1] = -l

    # Populate spatial/temporal disp. coords. 
    lT[:] = np.column_stack((l,T))

    # Clean-up
    del T, v, l
        
    # -------------------

    # Generate lookup-table: transect row/col coord. 
    x = dB0*np.arange(nB1)
    z = dB1*np.arange(nB0)

    # Generate look-up table: odometer wheel disp. 
    # (from its start point) 
    n = np.max(nA_B)                                           
    S = np.linspace(0,(dA*n),n)   

    # Generate look-up table: distance of antenna midpoint 
    # (relative to global datum) 
    R = -L/2 + np.arange(n)*(dA)

    # -------------------

    # Cycle transects...
    for i in range(nB):

        # recover no. traces in ith transect 
        nA_i = nA_B[i]

        # extract trace disp. values
        l = R[0:nA_i]

        # duplicate values for each sample within trace
        lR = np.repeat(l,nS_A)

        # indexes of samples in ith transect
        i0 = (nS_A)*np.sum(nA_B[0:i])
        i1 = i0 + (nS_A*nA_i)

        # if row...
        if (i < nB0):
            xyz[i0:i1,0] = lR
            xyz[i0:i1,2] = np.repeat(z[i],len(lR))
        else: 
            xyz[i0:i1,0] = np.repeat(x[i-nB0],len(lR))
            xyz[i0:i1,2] = lR

        # progress update
        progress(nB,i,'...')

    # -------------------
    return xyz, lT

# Functions to DIRECTLY Recover Amplitude Data 
def getA_gprmax(filePaths):
    """
    Returns temporal amplitude data directly from input files.  

    Parameters: 
    filePaths (array): List of relevant input files
    
    Returns:
    A (array): Amplitude values

    Notes: 

    collapse to 1D
        (-) recall amplitude array in hdf5 file has samples along 
            rows, traces along cols
        (-) <.flatten()> by default is row-major order 
            (i.e. flatten rows then columns)
        (-) as samples should be sequential, must use col-major 
            order instead
        (-) as fortran uses col-major order, the numpy 
            shorthand is 'F'
    

    """    

    # Recover global parameters
    nS_A, _ = getGlobals_gprmax(filePaths[0])
    nB = len(filePaths)

    # Recover local variables 
    nA_B, nA = getLocals_gprmax(filePaths)
    nS_B = nS_A*nA_B                                    
    nS = nS_A*nA    

    # Pre-allocate storage array
    A = np.zeros(nS)  

    # ------------------

    # Cycle transects...
    for i in range(nB):

        # import transect amplitude array
        with h5py.File(filePaths[i],'r') as h5:
            data = h5['rxs']['rx1']['Ez'][:]

        # collapse to 1D
        data = data.flatten(order='F')

        # populate storage
        i0 = 0 + i*nS_B
        i1 = i0 + nS_B
        A[i0:i1] = data 

    # -------------------
    return A
def getA_vna(filePaths):
    """
    Returns temporal amplitude data directly from input files.  

    Parameters: 
    filePaths (array): List of relevant input files
    
    Returns:
    A (array): Amplitude values

    """    

    # Recover global parameters
    nS_A, dt, fmt, f = getGlobals_vna(filePaths[0])
    nA = len(filePaths)

    # Recover local variables
    nA_B, nB = getLocals_vna(filePaths)
    nS_B = nS_A*nA_B
    nS = nS_B*nB

    # Pre-allocate storage
    s11 = np.zeros((nS,2))
    A = np.zeros((nS_A,nA))

    # -------------------

    # Cycle traces...
    for i in range(nA):

        # define sample indexes for ith trace
        i0 = 0 + i*nS_A
        i1 = i0 + nS_A

        # import s1p data
        with open(filePaths[i],'r') as file:
            s11[i0:i1,:] = np.loadtxt(file,
                                      skiprows=12,
                                      usecols=(1,2))

        # progress update
        progress(nA,i,'...')

    # Clean-up
    del filePaths, i0, i1

    # Convert s11 components to dimensionless-polar complex value
    if (fmt == 'DB'):
        r, t = s11[:,0], s11[:,1]
        r, t = 10**(r/20), np.deg2rad(t)
    elif(fmt == 'RI'):
        x, y = s11[:,0], s11[:,1]
        z = x + (1j)*y
        r = np.abs(z)
        t = np.angle(z)
        del x, y
    else: 
        raise Exception(f'invalid complex data format: {fmt}')
    Zf = r*np.exp(1j*t)
    
    # Clean-up
    del s11, r, t

    # Reshape s.t. samples along rows, traces along cols. 
    Zf =  Zf.reshape((nS_A,nA_B*nB),order='F')
    
    # -------------------

    # Define useful values
    pi = np.pi
    N = nS_A
    n = np.arange(N)

    # Set windowing function
    flag = 'kaiser_bessel'

    # Define windowing functions(s)
    if (flag == 'kaiser_bessel'):
        
        # parameters
        a0 = 3     

        # window function                             
        w_top = np.i0(a0*pi*np.sqrt(1-((2*n)/N - 1)**2))
        w_bot = np.i0(a0*pi)
        w = w_top/w_bot 
    elif (flag == 'rectangular'):
        
        # parameters
        # ~

        # window function
        w = np.ones(N)
    elif (flag == 'hamming'):
        
        # parameters 
        a0 = 25/46                                                                                 
        a1 = 1-a0

        # window function
        w = a0 - a1*np.cos((2*pi*n)/N) 
    elif (flag == 'hann'):
        
        # parameters
        a0 = 0.5                                                                                
        a1 = 1-a0

        # window function 
        w = a0 - a1*np.cos((2*pi*n)/N) 
    elif (flag == 'blackman'):
        
        # parameters -  'truncated' blackman
        #alpha = 0.16
        #a0 = (1-alpha)/2
        #a1 = 1/2
        #a2 = alpha/2

        # parameters - 'exact' blackman 
        a0 = 7938/18608
        a1 = 9240/18608
        a2 = 1430/18608

        # window function
        w = a0 - a1*np.cos((2*pi*n)/N) + a2*np.cos((4*pi*n)/N)
    elif (flag == 'blackman_harris'):
        
        # parameters
        a0=0.35875
        a1=0.48829
        a2=0.14128
        a3=0.01168

        # window function
        w = a0 - a1*np.cos((2*pi*n)/N) + a2*np.cos((4*pi*n)/N) \
            - a3*np.cos((6*pi*n)/N) 
    elif (flag == 'nuttal'):
        
        # parameters
        a0=0.355768
        a1=0.487396
        a2=0.144232
        a3=0.012604

        # window function 
        w = a0 - a1*np.cos((2*pi*n)/N) + a2*np.cos((4*pi*n)/N) \
            - a3*np.cos((6*pi*n)/N)
    elif (flag == 'turkey'):
        alpha = 0.5
        m0 = int((alpha*N)/2)
        m1 = int(N/2)
        w0 = (1/2)*(1 - np.cos((2*pi*n[0:m0])/(alpha*N)))
        w2 = w0[::-1]
        w1 = np.repeat(1,N-len(w2)-len(w0))
        w = np.concatenate((w0,w1,w2),axis=0)
    else:
        raise Exception(f'invalid window: {flag}')

    # Duplicate window function across all traces
    w = np.tile(w[:,np.newaxis],(1,nA_B*nB))

    # Apply window by convolution theorem
    Zf_w = w*Zf

    # Clean-up
    del pi, N, n, w, Zf

    # -------------------

    # recover t-domain amplitude values by IFFT
    Zt = np.fft.ifft(Zf_w, axis=0)

    # -------------------

    # Flatten s.t. sample by sample 
    A = Zt.flatten(order='F')
    del Zt

    # -------------------
    return A
def getA_zond(filePaths):
    """
    Returns temporal amplitude data directly from input files.  

    Parameters: 
    filePaths (array): List of relevant input files
    
    Returns:
    A (array): Amplitude values

    """    

    # Recover global parameters
    _, _, nb_H, nb_h, nS_A, nb_S, _, _ = \
        getGlobals_zond(filePaths[0])
    nB = len(filePaths)

    # Recover local variables
    nA_B, nA = getLocals_zond(filePaths)                          
    nb_A = nb_S*nS_A                                           
    nS = nS_A*nA                                              
    
    # Define sample data format
    if (nb_S == 2):
        dataType = 'int16'
    elif (nb_S == 4):
        dataType = 'int32'
    else: 
        raise Exception(f'unexpected no. bytes per trace: {nb_S}')

    # Pre-allocate storage array
    A = np.zeros(nS,dtype=dataType)                                     

    # ------------------

    # Initialise sample index counters
    k0 = 0
    k1 = k0 + nS_A

    # Cycle transects...
    for i in range(nB):

        # recover no. traces in ith transect 
        nA_i = nA_B[i]       

        # recover no. samples in ith transect
        nS_i = nS_A*nA_i                               

        # cycle traces...
        for j in range(nA_i):

            # compute starting byte index
            b0 = b0 = nb_H + j*(nb_h + nb_S*nS_A) + nb_h

            # import trace amplitude data
            A[k0:k1] = readBytes(filePaths[i],b0,dataType,nb_A)

            # update sample index counters 
            k0 += nS_A
            k1 += nS_A

        # progress update
        progress(nB,i,'...')
        
    # -------------------
    return A

# Function to Recover Spatial-Temporal Conversion Data
def getTlr_vna(dirPath):
    """
    Returns spatial/temporal disp.s of sample points

    Parameters: 
    dirPath (string): Current working directory filepath
    
    Returns:
    T (array): Temporal disp. (along 1D trace) from vna port 1 [s]
    l (array): Spatial disp. (along 1D trace) from vna port 1 [m]
    r (array): Radial disp. (along 3D boresight vector) from 
                centre of rotation (i.e. antenna base) [m]   

    Notes: 

    centre fo rotation is approx. antenna base

    piecewise approx to multi-layered disp. computation
        (-) antenna 'base' does not include coax connector
        (-) coax connector length is 13.65mm, recovered from 
            antenna schematics sent to UoD 
        (-) connector assumed comparable propagation velocity to vna 
            cable, hence vna cable 5m --> 5.01365m. 

    compute lift off distance by azimuth 
        (-) assumes non-curved contact surface with no significant 
            undulations 
        (-) assumes rotational scan setup s.t. symmetric about 
            vertical downward, with odd no. traces recorded 
            --> s.t. a vertical downward trace is always present 

    compute sample point disp.s - changes with lift-off distance 
        (-) r_lookup[:,:] rows denote sample point index within 
            a trace  
        (-) r_lookup[:,:] cols denote trace index within a transect 

    define layer extents
        (-) vna cable and antenna coaxial connector grouped as 
            single layer, owing to assumed similar electrical 
            characteristics
        (-) vna cable is 5m long, connector is 0.01365m long

    define layer propagation velocities
        (-) refractive index (n = c0/v) approx. 1 for air/free-space
        (-) from vna cable specification, propagation velocity 
            quoted at 81% speed of light in vacuum. 
    
    azimuth in degrees or radians 
        (-) azimuth is converted from degrees to radians first time 
            calculus is performed
            --> here, tigomometry to compute contact surface lift 
                off subject to azimuth      

    map s.t. r=0 aligns with rotation centre (i.e. antenna base)
        (-) necessary to associate disp. with disp. from more 
            'intuitive' rotation centre (rather than vna port)
        (-) -ve values therefore indicate sample points associated 
            with internal hardware reflection events  
        (-) -ve values replaced by nan's to avoid issues converting 
            to global coordinates in <xyz_vna()>         

    """

    # Recover metadata
    M = getMeta_vna(dirPath)

    # Unpack key metadata
    t0 = M[2]
    t1 = M[3]
    dt = M[5]
    scanType = M[6]
    epsilon_r = M[7]
    l_antenna = M[8]                                           
    l_down = M[9]                                              
    t_down = M[10]                                  

    # List trace files
    filePaths, _ = getFiles(os.path.join(dirPath,'traces'),['.s1p'])

    # Recover global parameters
    nS_A, dt, _, _ = getGlobals_vna(filePaths[0])

    # Recover local variables
    nA_B, nB = getLocals_vna(filePaths)
    nA = nA_B*nB

    # ------------------

    # Establish no. unique azimuth values (across full scan)
    if (scanType == 0):
        n = nA_B
    else:
        n = nB                        

    # Recover unique azimuth values 
    t = np.linspace(t0,t1,n)  

    # ------------------

    # define layer extents
    s0 = 5 + 0.01365                            
    s1 = l_antenna                              
    s2 = np.zeros(n)                            

    # compute lift-off distance by azimuth
    i_down = int(n/2) 
    for i in range(n):
        if (i < i_down):
            alpha = np.deg2rad(t_down - t[i])
            l_toRotationCentre = (l_antenna + l_down)/np.cos(alpha)
            l_liftOff = l_toRotationCentre - l_antenna
            s2[i] = l_liftOff
        elif (i > i_down):
            alpha = np.deg2rad(t[i] - t_down)
            l_toRotationCentre = (l_antenna + l_down)/np.cos(alpha)
            l_liftOff = l_toRotationCentre - l_antenna
            s2[i] = l_liftOff 
        elif (i == i_down):
            l_liftOff = l_down
            s2[i] = l_liftOff
        else:
            raise Exception(f'invalid azimuth: {t[i]}')
    del alpha, l_toRotationCentre, l_liftOff

    # ------------------

    # define layer propagation velocities
    v0 = 0.81*c0                                
    v1 = 0.81*c0                                 
    v2 = 1.0*c0                                 
    v3 = (1/(np.sqrt(epsilon_r)))*c0            

    # Compute 2-way travel time to reach each interface
    T0 = (2*s0)/v0                              
    T1 = T0 + (2*s1)/v1                        
    T2 = T1 + (2*s2)/v2                        

    # Compute 2-way travel time for each sample in trace
    T = dt*(np.arange(0,nS_A))

    # partition sample indexes by layer
    n0 = len(T[(T>=0) & (T<T0)])                
    n1 = len(T[(T>=T0) & (T<T1)])               
    n2 = np.zeros(n)
    n3 = np.zeros(n)
    for i in range(n):                          
        n2[i] = len(T[(T>=T1) & (T<T2[i])])
        n3[i] = len(T[T>=T2[i]])                 

    # ------------------
    
    # Define storage array
    l_lookup = np.zeros((nS_A,n))

    # Compute sample point disp.s (from vna) - same for all traces
    l0 = (v0/2)*(dt*np.arange(0,n0))
    l1 = l0[-1] + (v1/2)*(dt*(np.arange(0,n1)+1))

    # Compute sample point disp.s - changes with lift-off distance 
    for i in range(n):
        l2 = l1[-1] + (v2/2)*(dt*(np.arange(0,n2[i])+1))
        l3 = l2[-1] + (v3/2)*(dt*(np.arange(0,n3[i])+1))
        l_lookup[:,i] = np.concatenate((l0,l1,l2,l3),axis=0)

    # ------------------

    # Consolidate spatial disp.s from vna (sample by sample)
    # note: currently, r=0 aligns with vna port 1
    if (scanType == 0):
        l_B = l_lookup.flatten(order='F')
        l = np.tile(l_B,nB)
    else:
        l_B = np.repeat(l_lookup,nA_B,axis=1)
        l = l_B.flatten(order='F')

     # map s.t. r=0 aligns with rotation centre (i.e. antenna base)
    iS_T0 = np.where(T>T0)[0][0]        
    l_T0 = l[iS_T0]                     
    r = l - l_T0                                                                               

    # Convert -ve radial values to <nan>
    r[r<0] = np.nan

    # ------------------

    # Consolidate temporal disp.s from vna (sample by sample)
    T = np.tile(T,nA)

    # ------------------
    return T, l, r

# Functions to Plot Scan Trajectories
def plotTraj_gprmax(dirPath,xyz,iSAB):
    """
    Returns scan trajectory plot. 

    Parameters: 
    dirPath (string): Current working directory filepath
    xyz (array): Cartesian coords. 
    iSAB (array): sample/trace/transect indexes 
    
    Returns:
    None

    Notes: 

    define dark/light trace colouring
        (-) for a given trace, sample advance indicated by colour 
            contrast (dark --> light) 
        (-) for a given transect, trace advance indicated by 
            colour hue (red --> green -> blue)
        (-) for a given grid scan, transect advance indicated 
            by separation of transect planes (low --> high)

    """

    # Recover metadata 
    M = getMeta_gprmax(dirPath)
    nB0 = int(M[0]) 
    nB1 = int(M[1])

    # Recover other useful values     
    nA_B = np.max(iSAB[:,1]) + 1
    nS_A = np.max(iSAB[:,0]) + 1 

    # ------------------
    
    # Define transects of interest             
    BoI = []                                    
    if True:                # add transects - gird scan rows
        BoI.extend([0,9,(nB0-1)])                
    if True:                # add transects - grid scan cols
        BoI.extend([nB0,nB0+9,(nB0-1)+(nB1-1)])   
    BoI = np.array(BoI)  
    nBoI = len(BoI)                                                       
      
    # Define traces of interest
    # recall: trace indexes start from 0 (thus -1)
    AoI = np.array([0,9,(nA_B-1)]) 
    nAoI = len(AoI) 
    
    # Allocate storage 
    # note: [ samples | xyz | transect # | trace # ]
    X = np.zeros((nS_A,3,nBoI,nAoI))    

    # Extract position data                                                            
    for i in range(nBoI):
        for j in range(nAoI):
            X[:,:,i,j] = xyz[(iSAB[:,2] == BoI[i]) & \
                             (iSAB[:,1] == AoI[j])]

    # ------------------

    # Configure plot style 
    plt.rcParams.update({'font.size': 10, 
                         'font.family': 'serif', 
                         'font.serif': 'Times New Roman'}) 

    # Configure light/dark colour scale (for each trace)
    u = np.linspace(0.75, 0.25, nS_A)
    c = np.zeros((nS_A,4,3))
    colourMaps = ['Reds','Greens','Blues']
    for i in range(len(colourMaps)):
        cmap = plt.get_cmap(colourMaps[i])
        c[:,:,i] = cmap(u)
    del u, colourMaps, cmap

    # ------------------

    # Create figure 
    fig0 = plt.figure(figsize=(6, 4))  
    ax0 = fig0.add_subplot(111,projection='3d',proj_type='ortho')  

    # Set axis limits
    # x0, y0, z0 = np.min(xyz,axis=0)
    # x1, _, z1 = np.max(xyz,axis=0)
    if False:
        if (x0 > 0): x0 = 0
        if (z0 > 0): z0 = 0
        if (x1 < 0): x1 = 0
        if (z1 < 0): z1 = 0
        ax0.set_xlim(x0,x1)
        ax0.set_ylim(z0,z1)
    
    Lx = 0.420
    Ly = 0.450
    Lz = 0.420
    
    ax0.set_xlim(-0.05,0.50)
    ax0.set_ylim(-0.05,0.50)
    ax0.set_zlim(-0.5,0.05)

    # Set aspect ratio
    ax0.set_box_aspect([1,1,1]) 

    # Highlight XZ plane
    if False:
        if True:                        
            Lx = 0.42021                
            Lz = 0.4199
            x0 = 0; x1 = Lx
            z0 = 0; z1 = Lz
            ax0.set_xlim(x0,x1)
            ax0.set_ylim(z0,z1)
        x = np.linspace(x0,x1,100)
        z = np.linspace(z0,z1,100)
        x, z = np.meshgrid(x, z)
        y = np.zeros_like(x)
        ax0.plot_surface(x, z, y, color='grey', alpha=0.10)
        del x,z,y

    # Highlight origin
    if False:
        s = 0.25
        ax0.plot([0, s*x1], [0, 0], [0, 0], color='k',alpha=0.8)           
        ax0.plot([0, 0], [0, s*z1], [0, 0], color='k',alpha=0.8)      
        ax0.plot([0, 0], [0, 0], [s*y0, 0], color='k',alpha=0.8)      
        del s

    # Show RoI outline
    if True: 

        def cuboid_data(origin, size):
            # this function returns the vertices of a cuboid
            l, w, h = size
            x = [origin[0], origin[0] + l, origin[0] + l, \
                 origin[0], origin[0], origin[0] + l, origin[0] \
                    + l, origin[0]]
            y = [origin[1], origin[1], origin[1] + w, origin[1] \
                 + w, origin[1], origin[1], origin[1] + w, \
                    origin[1] + w]
            z = [origin[2], origin[2], origin[2], origin[2], \
                 origin[2] + h, origin[2] + h, origin[2] + h, \
                    origin[2] + h]
            return x, y, z
        
        def plot_cuboid(ax, origin, size):
            x, y, z = cuboid_data(origin, size)
            # define vertices for faces
            vertices = [
                [[x[0], y[0], z[0]], [x[1], y[1], z[1]], 
                 [x[5], y[5], z[5]], [x[4], y[4], z[4]]],  
                [[x[2], y[2], z[2]], [x[3], y[3], z[3]], 
                 [x[7], y[7], z[7]], [x[6], y[6], z[6]]], 
                [[x[0], y[0], z[0]], [x[1], y[1], z[1]], 
                 [x[2], y[2], z[2]], [x[3], y[3], z[3]]],  
                [[x[4], y[4], z[4]], [x[5], y[5], z[5]], 
                 [x[6], y[6], z[6]], [x[7], y[7], z[7]]],  
                [[x[0], y[0], z[0]], [x[3], y[3], z[3]], 
                 [x[7], y[7], z[7]], [x[4], y[4], z[4]]], 
                [[x[1], y[1], z[1]], [x[2], y[2], z[2]], 
                 [x[6], y[6], z[6]], [x[5], y[5], z[5]]]  
            ]
            faces = Poly3DCollection(vertices, linewidths=1, 
                                     edgecolors='k')
            faces.set_facecolor((0.5, 0.5, 0.5, 0.1))  
            ax.add_collection3d(faces)

        origin = [0, 0, 0]
        size = [Lx, Lz, -Lz]  
        plot_cuboid(ax0, origin, size)

    # Show targets
    if True: 

        def plot_Sphere(ax, center, radius, opacity):
            
            # generate sphere data 
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = center[2] + radius * np.outer(np.ones(np.size(u)), 
                                              np.cos(v))
            
            # add to plot
            ax.plot_surface(x, y, z, color='grey', alpha=opacity)

        plot_Sphere(ax0, [0.210,0.237,-0.210], 0.1/2, 0.5) 
        plot_Sphere(ax0, [0.272,0.148,-0.210], 0.06/2, 0.5) 
        plot_Sphere(ax0, [0.147,0.148,-0.210], 0.03/2, 0.5) 

    # Toggle view/axes
    if True:
        show_iso = False
        show_xy = False
        show_xz = True
        show_yz = False
        if (show_iso == True):
            ax0.view_init(elev=30, azim=30, roll=0)
            ax0.set_xlabel('x') 
            ax0.set_ylabel('z')
            ax0.set_zlabel('y')  
            plt.tight_layout()
        elif (show_xy == True):
            ax0.view_init(elev=0, azim=-90, roll=0)
            ax0.set_xlabel('x') 
            ax0.set_zlabel('y')  
            ax0.set_yticklabels([])
        elif (show_xz == True):
            ax0.view_init(elev=90, azim=-90, roll=0)
            ax0.set_xlabel('x') 
            ax0.set_ylabel('z')
            ax0.set_zticklabels([])
        elif (show_yz == True):
            ax0.view_init(elev=0, azim=-0, roll=0) 
            ax0.set_ylabel('z')
            ax0.set_zlabel('y')  
            ax0.set_xticklabels([])
        del show_iso, show_xy, show_xz, show_yz

    # ------------------

    # Add traces(A)/transects(B) of interest
    if True:
        cN = np.shape(c)[2]
        for i in range(len(BoI)):
            for j in range(len(AoI)):
                ax0.scatter(X[1:,0,i,j], X[1:,2,i,j], 
                            X[1:,1,i,j],color=c[1:,:,j%cN],
                            marker='.')        
                ax0.scatter(X[0,0,i,j], X[0,2,i,j], 
                            X[0,1,i,j],color=c[0,:,j%cN],
                            marker='.')             

    # ------------------

    # Set white background for the figure and the plot area
    fig0.patch.set_facecolor('white') 
    ax0.set_facecolor('white')        

    # Export 
    if True:
        dirPath = os.path.dirname(os.path.abspath(__file__))
        fig0.savefig(os.path.join(dirPath,'output.png'), dpi=350) 
        fig0.savefig(os.path.join(dirPath,'output.pdf'))          
        fig0.savefig(os.path.join(dirPath,'output.svg'))     

    # Display
    plt.show()

    # ------------------
    return [] 
def plotTraj_zond(dirPath,xyz,iSAB):
    """
    Returns scan trajectory plot. 

    Parameters: 
    dirPath (string): Current working directory filepath
    xyz (array): Cartesian coords. 
    iSAB (array): sample/trace/transect indexes 
    
    Returns:
    None

    """

    # Recover metadata 
    M = getMeta_zond(dirPath)
    nB0 = int(M[0]) 
    nB1 = int(M[1])

    # Recover other useful values
    filePaths, _ = getFiles(
        os.path.join(dirPath,'transects'),['.sgy'])
    nA_B, _ = getLocals_zond(filePaths) 
    _, _, _, _, nS_A, _, _, _ = getGlobals_zond(filePaths[0])

    # ------------------

    # Define transects of interest             
    BoI = []                                    
    if True:                
        BoI.extend([0,9,(nB0-1)])                
    if True:               
        BoI.extend([nB0,nB0+9,(nB0-1)+(nB1-1)])                                                            
    BoI = np.array(BoI)
    nBoI = len(BoI)

    # Define traces of interest
    # recall: trace indexes start from 0 (thus -1)
    AoI = np.zeros((nBoI,3),dtype=np.int16)
    AoI[:,1] = 9
    for i in range(nBoI):
        AoI[i,2] = nA_B[BoI[i]] - 1
    nAoI = len(AoI) 

    # Allocate storage
    # note: [ samples | xyz | transect # | trace # ]
    X = np.zeros((nS_A,3,nBoI,nAoI))            

    # Extract position data                                                            
    for i in range(nBoI):      
        for j in range(nAoI):                  
            X[:,:,i,j] = xyz[(iSAB[:,2] == BoI[i]) & \
                             (iSAB[:,1] == AoI[i,j%3])]

    # ------------------

    # Configure plot style 
    plt.rcParams.update({'font.size': 10, 
                         'font.family': 'serif', 
                         'font.serif': 'Times New Roman'}) 

    # Configure light/dark colour scale (for each trace)
    u = np.linspace(0.75, 0.25, nS_A)
    c = np.zeros((nS_A,4,3))
    colourMaps = ['Reds','Greens','Blues']
    for i in range(len(colourMaps)):
        cmap = plt.get_cmap(colourMaps[i])
        c[:,:,i] = cmap(u)
    del u, colourMaps, cmap

    # ------------------

    # Create figure 
    fig0 = plt.figure(figsize=(6, 4))  
    ax0 = fig0.add_subplot(111,projection='3d',proj_type='ortho')  

    # Set axis limits
    # x0, y0, z0 = np.min(xyz,axis=0)
    # x1, _, z1 = np.max(xyz,axis=0)
    if False:
        if (x0 > 0): x0 = 0
        if (z0 > 0): z0 = 0
        if (x1 < 0): x1 = 0
        if (z1 < 0): z1 = 0
        ax0.set_xlim(x0,x1)
        ax0.set_ylim(z0,z1)

    Lx = 0.66                   
    Ly = 0.52
    Lz = 0.53
    
    ax0.set_xlim(-0.2,0.90)
    ax0.set_ylim(-0.2,0.90)
    ax0.set_zlim(-0.55,0.05)

    # Set aspect ratio
    ax0.set_box_aspect([1,1,1]) 

    # Highlight origin
    if False:
        s = 0.25
        ax0.plot([0, s*x1], [0, 0], [0, 0], color='k',alpha=0.8)            
        ax0.plot([0, 0], [0, s*z1], [0, 0], color='k',alpha=0.8)     
        ax0.plot([0, 0], [0, 0], [s*y0, 0], color='k',alpha=0.8)      
        del s

    # Highlight XZ plane
    if False:
        if True:                        
            Lx = 0.66            
            Lz = 0.53
            x0 = 0
            z0 = 0
            ax0.set_xlim(x0,x1)
            ax0.set_ylim(z0,z1)
        x = np.linspace(x0,x1,100)
        z = np.linspace(z0,z1,100)
        x, z = np.meshgrid(x, z)
        y = np.zeros_like(x)
        ax0.plot_surface(x, z, y, color='grey', alpha=0.10)
        del x,z,y
 
    # Show RoI outline
    if True: 

        def cuboid_data(origin, size):
            # this function returns the vertices of a cuboid
            l, w, h = size
            x = [origin[0], origin[0] + l, origin[0] + l, \
                 origin[0], origin[0], origin[0] + l, origin[0] \
                    + l, origin[0]]
            y = [origin[1], origin[1], origin[1] + w, origin[1] \
                 + w, origin[1], origin[1], origin[1] + w, \
                    origin[1] + w]
            z = [origin[2], origin[2], origin[2], origin[2], \
                 origin[2] + h, origin[2] + h, origin[2] + h, \
                    origin[2] + h]
            return x, y, z
        
        def plot_cuboid(ax, origin, size):
            x, y, z = cuboid_data(origin, size)
            # define vertices for faces
            vertices = [
                [[x[0], y[0], z[0]], [x[1], y[1], z[1]], 
                 [x[5], y[5], z[5]], [x[4], y[4], z[4]]],  
                [[x[2], y[2], z[2]], [x[3], y[3], z[3]], 
                 [x[7], y[7], z[7]], [x[6], y[6], z[6]]], 
                [[x[0], y[0], z[0]], [x[1], y[1], z[1]], 
                 [x[2], y[2], z[2]], [x[3], y[3], z[3]]],  
                [[x[4], y[4], z[4]], [x[5], y[5], z[5]], 
                 [x[6], y[6], z[6]], [x[7], y[7], z[7]]],  
                [[x[0], y[0], z[0]], [x[3], y[3], z[3]], 
                 [x[7], y[7], z[7]], [x[4], y[4], z[4]]],  
                [[x[1], y[1], z[1]], [x[2], y[2], z[2]], 
                 [x[6], y[6], z[6]], [x[5], y[5], z[5]]]   
            ]
            faces = Poly3DCollection(vertices, linewidths=1, 
                                     edgecolors='k')
            faces.set_facecolor((0.5, 0.5, 0.5, 0.1))  
            ax.add_collection3d(faces)

        origin = [0, 0, 0]
        size = [Lx, Lz, -0.52]  
        plot_cuboid(ax0, origin, size)

    # Show targets
    if True: 

        def plot_Sphere(ax, center, radius, opacity):
            
            # generate sphere data 
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = center[2] + radius * np.outer(np.ones(np.size(u)), 
                                              np.cos(v))
            
            # add to plot
            ax.plot_surface(x, y, z, color='grey', alpha=opacity)

        plot_Sphere(ax0, [0.386,0.265,-0.130], 0.1/2, 0.5)
        plot_Sphere(ax0, [0.253,0.213,-0.130], 0.06/2, 0.5) 
        plot_Sphere(ax0, [0.253,0.337,-0.130], 0.03/2, 0.5) 

    # Toggle view/axes
    if True:
        show_iso = False
        show_xy = False
        show_xz = True
        show_yz = False
        if (show_iso == True):
            ax0.view_init(elev=30, azim=30, roll=0)
            ax0.set_xlabel('x') 
            ax0.set_ylabel('z')
            ax0.set_zlabel('y')  
            plt.tight_layout()
        elif (show_xy == True):
            ax0.view_init(elev=0, azim=-90, roll=0)
            ax0.set_xlabel('x') 
            ax0.set_zlabel('y')  
            ax0.set_yticklabels([])
        elif (show_xz == True):
            ax0.view_init(elev=90, azim=-90, roll=0)
            ax0.set_xlabel('x') 
            ax0.set_ylabel('z')
            ax0.set_zticklabels([])
        elif (show_yz == True):
            ax0.view_init(elev=0, azim=-0, roll=0) 
            ax0.set_ylabel('z')
            ax0.set_zlabel('y')  
            ax0.set_xticklabels([])
        del show_iso, show_xy, show_xz, show_yz

    # ------------------

    # Add traces(A)/transects(B) of interest
    if True:
        cN = np.shape(c)[2]
        for i in range(nBoI):
            for j in range(nAoI):
                ax0.scatter(X[1:,0,i,j], X[1:,2,i,j], 
                            X[1:,1,i,j],color=c[1:,:,j%cN],
                            marker='.')       
                ax0.scatter(X[0,0,i,j], X[0,2,i,j], X[0,1,i,j],
                            color=c[0,:,j%cN],marker='.')           

    # ------------------

    # Set white background for the figure and the plot area
    fig0.patch.set_facecolor('white')  
    ax0.set_facecolor('white')       

    # Export 
    if True:
        dirPath = os.path.dirname(os.path.abspath(__file__))
        fig0.savefig(os.path.join(dirPath,'output.png'), dpi=350) 
        fig0.savefig(os.path.join(dirPath,'output.pdf'))          
        fig0.savefig(os.path.join(dirPath,'output.svg'))     

    # Display
    plt.show()

    # ------------------
    return []
def plotTraj_vna(dirPath,xyz,iSAB):
    """
    Returns scan trajectory plot. 

    Parameters: 
    dirPath (string): Current working directory filepath
    xyz (array): Cartesian coords. 
    iSAB (array): sample/trace/transect indexes 
    
    Returns:
    None

    """

    # Recover metadata 
    # ~ 

    # Recover other useful values
    nB = np.max(iSAB[:,2]) + 1
    nA_B = np.max(iSAB[:,1]) + 1
    nS_A = np.max(iSAB[:,0]) + 1 
    
    # ------------------

    # Define transects of interest                                                
    BoI = np.array([0,9,nB-1])  
    nBoI = len(BoI)                                                       
      
    # Define traces of interest
    AoI = np.array([0,9,(nA_B-1)]) 
    nAoI = len(AoI) 
    
    # Allocate storage 
    # note: [ samples | xyz | transect # | trace # ]
    X = np.zeros((nS_A,3,nBoI,nAoI))    

    # Extract position data                                                            
    for i in range(nBoI):
        for j in range(nAoI):
            X[:,:,i,j] = xyz[(iSAB[:,2] == BoI[i]) & \
                             (iSAB[:,1] == AoI[j])]

    # ------------------ 

    # Configure plot style 
    plt.rcParams.update({'font.size': 10, 
                         'font.family': 'serif', 
                         'font.serif': 'Times New Roman'}) 

    # Configure light/dark colour scale (for each trace)
    u = np.linspace(0.75, 0.25, nS_A)
    c = np.zeros((nS_A,4,3))
    colourMaps = ['Reds','Greens','Blues']
    for i in range(len(colourMaps)):
        cmap = plt.get_cmap(colourMaps[i])
        c[:,:,i] = cmap(u)
    del u, colourMaps, cmap

    # ------------------ 

    # Create figure 
    fig0 = plt.figure(figsize=(6, 4))  
    ax0 = fig0.add_subplot(111,projection='3d',proj_type='ortho')  

    # Define physical sample
    Lx = 0.66                   
    Ly = 0.69 + 0.53
    Lz = 0.53

    # Set axis limits
    # x0, y0, z0 = np.nanmin(xyz,axis=0)
    # x1, y1, z1 = np.nanmax(xyz,axis=0)
    if False:
        if (x0 > 0): x0 = 0
        if (z0 > 0): z0 = 0
        if (x1 < 0): x1 = 0
        if (z1 < 0): z1 = 0
        ax0.set_xlim(x0,x1)
        ax0.set_ylim(z0,z1)
    ax0.set_xlim(-0.3,0.80)
    ax0.set_ylim(-0.3,0.80)
    ax0.set_zlim(-0.6,0.80)

    # Set aspect ratio
    ax0.set_box_aspect([1,1,1]) 

    # Restrict view to physical sample
    # recall: traces extend beyond the physical sample 
    # if False:    
    #     z0 = -0.5; z1 = 0.5         
    #     ax0.set_ylim(z0,z1)
    #     ax0.set_zlim(-Ly,y1)                  

    # Highlight XZ plane
    if False:
        if True:                
            x0 = 0; x1 = Lx
            z0 = 0; z1 = Lz
        x = np.linspace(x0,x1,100)
        z = np.linspace(z0,z1,100)
        x, z = np.meshgrid(x, z)
        y = np.zeros_like(x)
        ax0.plot_surface(x, z, y, color='grey', alpha=0.10)
        del x,z,y

    # Show RoI outline
    if True: 

        def cuboid_data(origin, size):
            # this function returns the vertices of a cuboid
            l, w, h = size
            x = [origin[0], origin[0] + l, origin[0] + l, \
                 origin[0], origin[0], origin[0] + l, \
                    origin[0] + l, origin[0]]
            y = [origin[1], origin[1], origin[1] + w, \
                 origin[1] + w, origin[1], origin[1], \
                    origin[1] + w, origin[1] + w]
            z = [origin[2], origin[2], origin[2], \
                 origin[2], origin[2] + h, origin[2] + h,\
                    origin[2] + h, origin[2] + h]
            return x, y, z
        
        def plot_cuboid(ax, origin, size):
            x, y, z = cuboid_data(origin, size)
            # define vertices for faces
            vertices = [
                [[x[0], y[0], z[0]], [x[1], y[1], z[1]], 
                 [x[5], y[5], z[5]], [x[4], y[4], z[4]]],  
                [[x[2], y[2], z[2]], [x[3], y[3], z[3]], 
                 [x[7], y[7], z[7]], [x[6], y[6], z[6]]], 
                [[x[0], y[0], z[0]], [x[1], y[1], z[1]], 
                 [x[2], y[2], z[2]], [x[3], y[3], z[3]]], 
                [[x[4], y[4], z[4]], [x[5], y[5], z[5]], 
                 [x[6], y[6], z[6]], [x[7], y[7], z[7]]], 
                [[x[0], y[0], z[0]], [x[3], y[3], z[3]], 
                 [x[7], y[7], z[7]], [x[4], y[4], z[4]]],  
                [[x[1], y[1], z[1]], [x[2], y[2], z[2]], 
                 [x[6], y[6], z[6]], [x[5], y[5], z[5]]]   
            ]
            faces = Poly3DCollection(vertices, 
                                     linewidths=1, edgecolors='k')
            faces.set_facecolor((0.5, 0.5, 0.5, 0.1)) 
            ax.add_collection3d(faces)

        origin = [0, 0, 0]
        size = [Lx, Lz, -0.52] 
        plot_cuboid(ax0, origin, size)

    # Show targets
    if True: 

        def plot_Sphere(ax, center, radius, opacity):
            
            # generate sphere data 
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = center[2] + radius * np.outer(np.ones(np.size(u)), 
                                              np.cos(v))
            
            # add to plot
            ax.plot_surface(x, y, z, color='grey', alpha=opacity)

        plot_Sphere(ax0, [0.386,0.265,-0.130], 0.1/2, 0.5) 
        plot_Sphere(ax0, [0.253,0.213,-0.130], 0.06/2, 0.5) 
        plot_Sphere(ax0, [0.253,0.337,-0.130], 0.03/2, 0.5) 

    # Highlight origin
    if False:
        s = 0.25
        ax0.plot([0, s*x1], [0, 0], [0, 0], color='k',alpha=0.8)          
        ax0.plot([0, 0], [0, s*z1], [0, 0], color='k',alpha=0.8)      
        ax0.plot([0, 0], [0, 0], [s*y0, 0], color='k',alpha=0.8)      
        del s

    # Toggle view/axes
    if True:
        show_iso = False
        show_xy = False
        show_xz = True
        show_yz = False
        if (show_iso == True):
            ax0.view_init(elev=30, azim=30, roll=0)
            ax0.set_xlabel('x') 
            ax0.set_ylabel('z')
            ax0.set_zlabel('y') 
            plt.gca().xaxis.set_major_formatter(
                plt.FormatStrFormatter('%04.2f'))
            plt.gca().yaxis.set_major_formatter(
                plt.FormatStrFormatter('%04.2f')) 
            plt.gca().zaxis.set_major_formatter(
                plt.FormatStrFormatter('%04.2f')) 
            plt.tight_layout()
        elif (show_xy == True):
            ax0.view_init(elev=0, azim=-90, roll=0)
            ax0.set_xlabel('x') 
            ax0.set_zlabel('y')  
            plt.gca().xaxis.set_major_formatter(
                plt.FormatStrFormatter('%04.2f'))
            plt.gca().zaxis.set_major_formatter(
                plt.FormatStrFormatter('%04.2f')) 
            ax0.set_yticklabels([])
        elif (show_xz == True):
            ax0.view_init(elev=90, azim=-90, roll=0)
            ax0.set_xlabel('x') 
            ax0.set_ylabel('z')
            plt.gca().xaxis.set_major_formatter(
                plt.FormatStrFormatter('%04.2f'))
            plt.gca().yaxis.set_major_formatter(
                plt.FormatStrFormatter('%04.2f')) 
            ax0.set_zticklabels([])
        elif (show_yz == True):
            ax0.view_init(elev=0, azim=-0, roll=0) 
            ax0.set_ylabel('z')
            ax0.set_zlabel('y')  
            plt.gca().yaxis.set_major_formatter(
                plt.FormatStrFormatter('%04.2f')) 
            plt.gca().zaxis.set_major_formatter(
                plt.FormatStrFormatter('%04.2f')) 
            ax0.set_xticklabels([])
        del show_iso, show_xy, show_xz, show_yz

    # ------------------

    # Add traces(A)/transects(B) of interest
    if True:
        cN = np.shape(c)[2]
        for i in range(len(BoI)):
            for j in range(len(AoI)):
                mask = (X[1:,2,i,j] > 0) & ((X[1:,2,i,j] < Lz))
                n = sum(mask)
                alphas = np.exp(-0.1 * np.arange(n))[::-1]
                ax0.scatter(X[1:,0,i,j][mask], X[1:,2,i,j][mask], 
                            X[1:,1,i,j][mask],color=c[0:n,:,j%cN],
                            marker='.',alpha=alphas)        
                ax0.scatter(X[0,0,i,j], X[0,2,i,j], X[0,1,i,j],
                            color=c[0,:,j%cN],marker='.')                     

    # ------------------

    # Set white background for the figure and the plot area
    fig0.patch.set_facecolor('white')  
    ax0.set_facecolor('white')         

    # Export 
    if True:
        dirPath = os.path.dirname(os.path.abspath(__file__))
        fig0.savefig(os.path.join(dirPath,'output.png'), dpi=350) 
        fig0.savefig(os.path.join(dirPath,'output.pdf'))          
        fig0.savefig(os.path.join(dirPath,'output.svg'))        

    # Display
    plt.show()

    # ------------------ 
    return[]

# ---------------------------------------------------------
# PRE-PROCESSING
# ---------------------------------------------------------

# Function to Add Time Variable Gain
def addGain(A,lT):
    """
    Applies time-power gain (trace by trace)
    Adapted from GPRPy <tpowGain()> 
    see https://doi.org/10.1190/tle39050332.1   

    Parameters: 
    A (array): Amplitude values
    lT (array): Spatial-temporal disp.s

    
    Returns:
    A (array): Updated

    """

    update('adding gain...')

    # Extract two way travel time
    t = lT[:,1]                

    # Apply time-power gain
    if False:
        alpha = 1 #2                   # scaling exponent           
        G = t**alpha                # gain function

    # Apply exponential gain
    if True:
        alpha = 1 #1.5                   # scaling exponent
        G = np.exp(alpha*t)         # gain function 

    # Apply gain function
    A = G*A                         

    # ------------------
    return A

# Function to Remove Mean Trace Amplitude
def removeMeanTrace(A,iSAB):
    """
    Subtracts average trace/background response of full 3D domain 
    (trace by trace) 
    Adapted from GPRPy <removeMeanTrace()> 
    see https://doi.org/10.1190/tle39050332.1 
    
    Parameters: 
    A (array): Amplitude values
    iSAB (array): sample/trace/transect indexes
    
    Returns:
    A (array): Updated

    """
    
    update('removing mean trace...')

    # Suppress global background --> subtract average trace 
    # (of all traces)
    if True: 

        # extract key values 
        nS = int(np.shape(iSAB)[0])
        nS_A= np.max(iSAB[:,0]) + 1
        nA = nS//nS_A

        # compute the mean trace
        A = np.reshape(A,[-1,nA],order='F')         
        A_mu = np.mean(A,axis=1)                    

        # subtract the mean trace
        A = A - A_mu[:, np.newaxis]                
        A = np.reshape(A,-1,order='F')             

    # ------------------
    return A

# Function to Apply Dewow
def addDewow(A,iSAB):
    """
    Applies dewow (trace by trace). 

    Parameters: 
    A (array): Amplitude values
    iSAB (array): sample/trace/transect indexes
    
    Returns:
    A (array): Updated

    """

    update('applying dewow...')

    # Extract useful values
    nS = int(np.shape(iSAB)[0])         # no. samples total 
    nS_A= np.max(iSAB[:,0]) + 1         # no. samples per trace
    nA = nS//nS_A                       # no. traces total

    # Define window size
    # note: ensure an odd value 
    # note: set >nS_A to span full domain
    n = 100001   
    # n = 31 

    # Check window length is odd valued 
    # s.t. midpoint is well defined               
    if (n%2 == 0):
        raise ValueError('ensure window length is odd valued.')                   

    # Make array columns traces
    A = np.reshape(A,[-1,nA],order='F')        

    # Apply dewow
    if (n >= nS_A):         
        A_mu = np.mean(A,axis=0)               
        A = A - A_mu                            
    elif (n > 1):         
        
        # append ghost points (zero padding)
        ghostMode = 'zero'
        if (ghostMode == 'zero'):
            Ag = np.pad(A, (((n-1)//2, (n-1)//2), (0, 0)), 
                        mode='constant', constant_values=0)
        elif (ghostMode == 'mirror'):
            Ag = np.pad(A, (((n-1)//2, (n-1)//2), (0, 0)), 
                        mode='symmetric') 
        else: 
            raise Exception(
                f'invalid ghostMode selected: {ghostMode}')
        del ghostMode

        # apply windowed moving average (sample by sample)
        Ac = np.cumsum(Ag,axis=0)                                          
        del Ag
        for i in range(nS_A):
            if (i > 0):                                                         
                A[i,:] = (1/n)*(Ac[i+n-1,:] - Ac[i-1,:])                           
            else:                                                               
                A[i,:] = (1/n)*(Ac[n-1,:] - 0)                                      
        del Ac
    else:
        raise Exception(f'invalid window selection: {n}')
    
    # Reset array shape
    A = np.reshape(A,-1,order='F')                       

    # ------------------
    return A

# ---------------------------------------------------------
# GRID REGULARISATION
# ---------------------------------------------------------

# Functions to Truncate 3D Dataset to Approximate RoI Volume
# note: data currently a regular curvilinear grid
def trim_vna(dirPath,xyz,rte,lT,A,iSAB):
    """
    Remove sample points outside RoI or that contain NaNs
    
    Parameters: 
    dirPath (string): Current working directory filepath
    xyz (array): Cartesian coords. 
    rte (array): Cylindrical coords. 
    lT (array): Spatial-temporal disp.s
.
    A (array): Amplitude values
    iSAB (array): sample/trace/transect indexes
    
    Returns:
    xyz (array): Updated
    rte (array): Updated 
    lT (array): Updated
    A (array): Updated
    iSAB (array): Updated

    """    

    update('trimming dataset to RoI...')

    # Recover RoI extent
    M = getMeta_vna(dirPath)
    Lx = M[11]
    Ly = M[12]
    Lz = M[13]
    del M

    # Remove any samples with NaN values
    # recall: xyz coords are global, datum aligned to RoI
    # i.e. top corner of sample box
    mask = np.isnan(xyz).any(axis=1)
    if any(mask):
        xyz = xyz[~mask]
        rte = rte[~mask]
        lT = lT[~mask]
        A = A[~mask]
        iSAB = iSAB[~mask]

    # Remove sample points outside RoI in x
    update('--> in x...')
    mask = (xyz[:, 0] < 0) | (xyz[:,0] > Lx)
    if any(mask):
        xyz = xyz[~mask]
        rte = rte[~mask]
        lT = lT[~mask]
        A = A[~mask]
        iSAB = iSAB[~mask]

    # Remove sample points outside RoI in y
    # note: remove points in the air-gap region 
    update('--> in y...')
    mask = (xyz[:,1] > 0) | (xyz[:, 1] < -Ly)
    if any(mask):
        xyz = xyz[~mask]
        rte = rte[~mask]
        lT = lT[~mask]
        A = A[~mask]
        iSAB = iSAB[~mask]

    # Remove sample points outside RoI in z
    update('--> in z...')
    mask = (xyz[:,2] < 0) | (xyz[:,2] > Lz)
    if any(mask):
        xyz = xyz[~mask]
        rte = rte[~mask]
        lT = lT[~mask]
        A = A[~mask]
        iSAB = iSAB[~mask]

    # ------------------
    return xyz,rte,lT,A,iSAB
def trim_zond(dirPath,xyz,lT,A,iSAB):
    """
    Clips sample points outside RoI
    
    Parameters: 
    dirPath (string): Current working directory filepath
    xyz (array): Cartesian coords. 
    lT (array): Spatial-temporal disp.s
.
    A (array): Amplitude values
    iSAB (array): sample/trace/transect indexes
    
    Returns:
    xyz (array): Updated
    lT (array): Updated
    A (array): Updated
    iSAB (array): Updated

    """

    update('trimming dataset to RoI...')

    # Recover RoI extent
    M = getMeta_zond(dirPath)
    Lx = M[7]
    Ly = M[8]
    Lz = M[9]
    del M

    # Remove sample points outside RoI in x
    update('--> in x...')
    mask = (xyz[:, 0] < 0) | (xyz[:,0] > Lx)
    if any(mask):
        xyz = xyz[~mask]
        lT = lT[~mask]
        A = A[~mask]
        iSAB = iSAB[~mask]

    # Remove sample points outside RoI in y
    update('--> in y...')
    mask = (xyz[:, 1] > 0) | (xyz[:, 1] < -Ly)
    if any(mask):
        xyz = xyz[~mask]
        lT = lT[~mask]
        A = A[~mask]
        iSAB = iSAB[~mask]

    # Remove sample points outside RoI in z
    update('--> in z...')
    mask = (xyz[:, 2] < 0) | (xyz[:,2] > Lz)
    if any(mask):
        xyz = xyz[~mask]
        lT = lT[~mask]
        A = A[~mask]
        iSAB = iSAB[~mask]

    # ------------------
    return xyz,lT,A,iSAB

# Function to Recover Regularised 3D Grid of Sample Points
# note: data currently an irregular grid
# note: data will become a regular rectilinear grid
def getGrid(xyz, A, technology):
    """
    Returns regular query point grid from curvilinear grid.
    
    Parameters: 
    xyz (array): Cartesian coords. 
    A (array): Amplitude values
    technology (string): type of data collection technology used
                                'gprmax', 'zond', 'vna'
    
    Returns:
    Qx (array): regularised query point cartesian coords. 
    Qy (array): = 
    Qz (array): = 
    QA (array): regularised query point amplitude values

    """

    if ((technology == 'gprmax') | (technology == 'vna') | \
        (technology == 'zond')):

        # ------------------
        update('regularising 3D data grid...')

        # ------------------
        # Create 3D Regular Grid 
        # ------------------
        update('--> creating query grid...')

        # define no. bins in each dimension
        num_x_bins = len(np.unique(xyz[:,0]))
        num_y_bins = len(np.unique(xyz[:,1]))
        num_z_bins = len(np.unique(xyz[:,2]))

        # define the grid boundaries
        # (i.e. spatial extremals of sample points)
        x0, y0, z0 = np.min(xyz,axis=0)
        x1, y1, z1 = np.max(xyz,axis=0)

        # create the grid edges
        x_edges = np.linspace(x0, x1, num_x_bins + 1)
        y_edges = np.linspace(y0, y1, num_y_bins + 1)
        z_edges = np.linspace(z0, z1, num_z_bins + 1)

        # ------------------
        # Digitize the Sample Points to Find the Corresponding Bins
        # ------------------
        update('--> binning sample points...')

        # digitise sample points into bins 
        # (i.e. return the indices of the bins to which each 
        # sample point belongs)
        x_indices = np.digitize(xyz[:, 0], x_edges) - 1
        y_indices = np.digitize(xyz[:, 1], y_edges) - 1
        z_indices = np.digitize(xyz[:, 2], z_edges) - 1

        # clip to ensure indices are within bounds
        # (sample points near domain extremals fall negligible 
        # distances outside the binning domain)
        x_indices = np.clip(x_indices, 0, num_x_bins - 1)
        y_indices = np.clip(y_indices, 0, num_y_bins - 1)
        z_indices = np.clip(z_indices, 0, num_z_bins - 1)

        # combine the indices to get a unique bin index
        # (i.e. flatten multidimensional array, row-major order)
        bin_indices = x_indices + num_x_bins * \
            (y_indices + num_y_bins * z_indices)

        # ------------------
        # Accumulate Amplitude Values in Bins
        # ------------------
        update('--> accumulating binned amplitudes...')


        # initialize arrays to hold the sum of amplitudes and the 
        # count of points in each bin
        num_bins = num_x_bins * num_y_bins * num_z_bins
        bin_counts = np.zeros(num_bins)
        if (technology == 'vna'):
            bin_sums = np.zeros(num_bins, dtype=complex)
        else:
            bin_sums = np.zeros(num_bins)        
       
        # accumulate the amplitude sums and counts
        np.add.at(bin_sums, bin_indices, A)              
        np.add.at(bin_counts, bin_indices, 1)

        # ------------------
        # Compute Mean Amplitudes in Each Bin
        # ------------------
        update('--> averaging binned amplitudes...')

        # avoid division by zero
        nonZero_bins = bin_counts > 0
        mean_amplitudes = np.zeros_like(bin_sums)
        mean_amplitudes[nonZero_bins] = bin_sums[nonZero_bins] / \
            bin_counts[nonZero_bins]

        # reshape to the original query point grid shape
        mean_amplitudes_grid = mean_amplitudes.reshape(
            (num_x_bins, num_y_bins, num_z_bins))
        QA = mean_amplitudes_grid

        # ------------------
        # Recover Spatial Coords.
        # ------------------
        update('--> recovering spatial coords...')

        # recover query point grid xyz coordinates
        # (3D arrays where each element represents the coordinate 
        # of a point in the grid)
        x_coords = np.linspace(x0, x1, num_x_bins) 
        y_coords = np.linspace(y0, y1, num_y_bins)  
        z_coords = np.linspace(z0, z1, num_z_bins)   
        Qx, Qy, Qz = np.meshgrid(x_coords, y_coords, z_coords, \
                                 indexing='ij')
        del x_coords, y_coords,z_coords

        # reshape to 1D arrays (if needed)
        # QA = QA.flatten()
        # Qx = Qx.flatten()
        # Qy = Qy.flatten()
        # Qz = Qz.flatten()
        # Qxyz = np.column_stack((Qx,Qy,Qz))
    else: 
        raise Exception(f'invalid technology type: {technology}')

    # ------------------
    return Qx, Qy, Qz, QA

# ---------------------------------------------------------
# POST PROCESSING
# ---------------------------------------------------------

# Function to Apply 3D Stolt Migration
def addMigration(X,Y,Z,A):
    """
    Applies 3D Stolt frequency-wavenumber (fk) migration to 
    amplitude data
    
    Parameters: 
    X (array): Cartesian coords. 
    Y (array): =
    Z (array): = 
    A (array): Amplitude values
    
    Returns:
    A_migrated (array): updated amplitude values. 

    """
    
    update('applying fk-migration...')

    # Step 1: fourier transform data to fk-domain
    # Step 2: apply a suitable migration operator in fk-domain
    # Step 3: inverse fourier transform back to spatial domain
    # note:  windowing compensates for spectral leakage during IFFT
    # note: ensure data remains as 3D arrays

    # Relative electrical permittivity of sample
    # (~) in future, import via <getMeta_...()> 
    epsilon_r = 4       
    
    # ------------------
    
    update('--> apply fft...')

    # Apply discrete fourier transform (spatial --> fk domain)
    A_fft = fftn(A)

    # Centre the zero frequency component
    A_fft_shifted = fftshift(A_fft) 

    # ------------------
    
    update('--> apply migration operator...')

    # Recover amplitude array dimensions 
    nx, ny, nz = np.shape(A)

    # Define expected wave propagation velocity
    v = c0/np.sqrt(epsilon_r)                                                                  

    # Define wavenumber grids
    # recall: y is vertical spatial axis (i.e. depth)
    kx = np.fft.fftfreq(nx)
    ky = np.fft.fftfreq(ny)                                 
    kz = np.fft.fftfreq(nz)
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij') 

    # Compute angular frequency (\omega)
    pi = np.pi
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    omega = 2*pi*k*v

    # Compute migration operator
    m = np.exp(-1j * omega)                 # isotropic operator
    # m = np.exp(-1j * ky * Y)              # non-iso operator
    
    # Clean-up
    del kx, ky, kz, k, omega

    # Apply migration operator
    A_migrated_fft = A_fft_shifted * m

    # ------------------
    
    update('--> apply ifft...')

    # Perform inverse 3D ftt (fk domain--> spatial domain)  
    # note: output is now complex valued
    A_migrated_fft_unshifted = ifftshift(A_migrated_fft)
    A_migrated = ifftn(A_migrated_fft_unshifted)

    # ------------------
    return A_migrated

# Function to Apply 3D Gaussian Smoothing
def addGaussianSmoothing(A):
    """
    Applies gaussian smoothing filter to a 3D dataset of 
    amplitude values.

    Parameters: 
    A (array): Amplitude values
    
    Returns:
    A (array): updated amplitude values. 

    """

    update('applying gaussian smoothing...')

    # Define gaussian kernel
    sigma = np.array([1,1,1])           
    m = 'wrap'                          
    t = 4                               

    # Apply gaussian smoothing 
    A = gaussian_filter(A, sigma, mode=m, truncate=t)

    # ------------------
    return A

# Function to Recover Magnitude Response
def getAbs(A):
    """
    Return absolute magnitude of amplitude data. 

    Parameters: 
    A (array): Amplitude values
    
    Returns:
    A (array): updated amplitude values. 

    """

    update('computing absolute amplitude values...')

    # Take absolute value(s) element-wise
    A = np.abs(A)

    # ------------------
    return A

# Function to Recover Normalised Response
def getNormalised(A):
    """
    Return globally normalised amplitude data (A \in [0,1])

    Parameters: 
    A (array): Amplitude values
    
    Returns:
    A (array): updated amplitude values. 

    """

    update('applying global amplitude normalisation...')

    # Define normalisation range (A \in [a,b])
    a = 0
    b = 1

    # Recover extremal amplitude values 
    # note: should be 0 and 1 respectively
    A0 = np.min(A)
    A1 = np.max(A)

    # Compute scaling factor
    s = (b-a)/(A1-A0)

    # Apply global normalisation
    A = a + s*(A-A0)

    # ------------------
    return A

# Function to Recover Thresholded Response
def getThreshold(X,Y,Z,A):
    """
    Removes any sample points with normalised amplitude outside 
    specified range. 

    Parameters: 
    X (array): cartesian coords. 
    Y (array): =
    Z (array): = 
    A (array): Amplitude values
    
    Returns:
    X (array): updated cartesian coords. 
    Y (array): =
    Z (array): = 
    A (array): updated amplitude values. 

    """

    update('applying amplitude threshold...')

    # Define threshold range 
    # recall: assuming incoming amplitudes normalised to [0,1]
    a = 0.55 # 0.50
    b = 0.85 # 0.80

    # Collapse to 1D
    A = A.flatten()
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()

    # Identify datapoints within desired range
    mask = ((A >= a) & (A <= b))
    
    # Remove datapoints outside desired range
    # note: collapses datapoints from 3D --> 1D arrays
    A = A[mask]
    X = X[mask]
    Y = Y[mask]
    Z = Z[mask]

    # ------------------
    return X, Y, Z, A

# ---------------------------------------------------------
# 3D SEGMENTATION
# ---------------------------------------------------------

# Function to Implement HDBSCAN proximal-clustering
def get3DClusters(dirPath,X,Y,Z,A):
    """
    Returns 3D spatial profiles (clusters) of suspected anomalies.
    First assesses by isovalue, then by relative proximity.  

    Parameters: 
    dirPath (string): Current working directory filepath
    X (array): Sample point X coords. 
    Y (array): Sample point Y coords. 
    Z (array): Sample point Z coords.
    A (array): Sample point amplitudes

    Returns:
    XYZ (array): Cartesian coords. 
    A (array): Amplitude array 
    I (array): Isovalue bin values
    V (array): Isovalue bin assignment indexes
    Ci (array): Cluster assignment index (with assigned isovalue)
    Cc (array): Sample point in cluster with this centroid XYZ coord.  
    """

    update('recovering 3D clusters...')

    # Extract no. sample points
    N = np.shape(A)[0]

    # Consolidate position coords. for efficiency
    XYZ = np.column_stack((X,Y,Z))
    del X, Y, Z

    # ------------------

    update('--> recover amplitude isovalues...')

    # Recover isovalue bins
    I = getIsoBins(A,method='ss')
    nI = len(I)

    # Allocate sample points to amplitude isovalue bins
    # note: -1 as bin indexes from 1 (not 0)
    V = np.digitize(A,I) - 1

    # ------------------

    update('--> running proximal clustering...')

    # Define min no. sample points in a well-defined cluster
    # note: QuickHull requires at least 4 points (min complexity 
    # shape is tetrahedron)
    nC = 175 #100 #10

    # Define clustering alpha value (i.e. smoothness/coarseness 
    # of clusters, default = 1)
    a = 1.00

    # Initialise proximal clustering
    # note: default min size is 5 
    C = HDBSCAN(min_cluster_size=nC, 
                alpha=a, 
                store_centers='centroid') 
    # C = HDBSCAN(min_cluster_size=nC,alpha=1)

    # Initialise storage array(s)
    n = np.shape(A)[0]
    C_index = np.zeros_like(A)
    C_centroid = np.full((n,3), np.nan)
    del n

    # Cycle amplitude isovalue bins
    for i in range(nI):

        # extract binned sample points
        mask = (V == i)

        # count number of sample points in ith isovalue bin
        n = np.sum(mask)

        # check if enough points for at least one 3D cluster,
        if (n >= nC):
    
            # form clusters by relative proximity 
            # i.e. of sample points in ith bin
            C_indexes = C.fit_predict(XYZ[mask])
            C_index[mask] = C_indexes
            
            # associate each sample point in cluster with the 
            # centroid coord. 
            A = C_indexes
            B = C.centroids_
            C_centroids = np.full((A.size, B.shape[1]), np.nan)
            m = (A >= 0)
            C_centroids[m,:] = B[A[m]]
            C_centroid[mask,:] = C_centroids

        else: 

            # if no cluster can be formed for a given set of 
            # sample points, 
            # cluster label --> 'unassigned' (i.e. -1)
            C_index[mask] = -1

        # update console
        progress(nI,i,'...')

    # Clean up
    del C
    
    # ------------------
    update('--> consolidating clusters...')

    # Convert cluster indexes to integers
    Ci = C_index.astype(int)
    Cc = C_centroid
    del C_index, C_centroid

    # Consolidate arrays
    # XYZAVC = np.column_stack((XYZ,A,V,C_index))
    # I = isovalue bin values [A0,...,A1] 
    # V = isovalue bin assignment index [0,1,...,]
    # C = cluster assignment index, [0,1,...]

    # ------------------

    # Export data (checkpointing)
    filePath = os.path.join(dirPath,'output.h5')
    h5Out(filePath,XYZ,'XYZ')
    h5Out(filePath,A,'A')
    h5Out(filePath,I,'I')
    h5Out(filePath,V,'V')
    h5Out(filePath,Ci,'Ci')
    h5Out(filePath,Cc,'Cc')

    # ------------------
    return XYZ, A, I, V, Ci, Cc

# Function to Recover Isovalue Bins
def getIsoBins(A,method='ss'):
    """
    # Returns isovalue bins defined either by: 
    # (i) ..... <manual> manually
    # (ii)..... <fd> Freedman-Diaconis rule 
    # (iii) ... <ss> Shimazaki_Shinomoto method

    Parameters: 
    A (array): Sample point amplitudes. 
    method (string): Method handle defined above. 

    Returns:
    A_bins (array): Bin index for each sample point.

    """

    update(f'--> computing isovalue bins (via {method})...')

    # Recover amplitude range
    A0 = np.min(A)
    A1 = np.max(A)
    N = len(A)

    # Compute no. bins
    if (method == 'manual'):
        nI = 100
    elif (method == 'fd'):
        
        # compute amplitude IQR
        q75, q25 = np.percentile(A, [75 ,25])
        IQR = q75 - q25

        # compute bin width via Freedman-Diaconis rule
        h = (2*IQR)/(np.cbrt(N))

        # calculate number of bins
        nI = int(np.ceil((A1 - A0) / h))
    elif (method == 'ss'):
    
        # ------------------

        def costMISE(A,k):
            # cost function for mean integrated squared error for 
            # a given no. bins k

            # ------------------

            # compute bin edges for k bins evenly spaced between 
            # min and max of data
            bin_edges = np.linspace(A.min(), A.max(), k+1)
            
            # compute histogram of data using the specified 
            # bin edges
            bin_counts, _ = np.histogram(A, bin_edges)
            
            # Number of bins
            k = len(bin_counts)
            
            # Calculate mean and variance of bin counts
            mean_count = np.mean(bin_counts)
            variance_count = np.var(bin_counts)
            
            # Calculate the cost function C(k) using 
            # Shimazaki and Shinomoto method
            C = (2 * mean_count - variance_count) / (N*(D/k)**2)

            # ------------------
            return C
        
        # ------------------

        # compute range of amplitude data 
        D = A1-A0  

        # compute the range of possible number of bins to evaluate
        bins_range = np.arange(1, int(np.sqrt(N)) + 1)

        # calculate cost function for each number of bins 
        # in bins_range
        # costs = np.array([costMISE(A,k) for k in bins_range])
        costs = []
        i = 0
        n = len(bins_range)
        for k in bins_range:
            cost = costMISE(A, k)
            costs.append(cost)
            progress(n,i,'...')
            i += 1
        costs = np.array(costs)
        del i,n
        
        # find index of the minimum cost function value
        # i.e. optimal number of bins
        nI = bins_range[np.argmin(costs)]
        
    else:
        raise Exception(f'invalid isovalue binning method {method}')

    # Compute bin values 
    A_bins = np.linspace(A0,A1,nI) 

    # ------------------
    return A_bins

# ---------------------------------------------------------
# RENDERING
# ---------------------------------------------------------

# Function to Plot 3D Visual Output
def plot3D(dirPath,XYZ,A,I,V,Ci,Cc,mode):
    """
    Returns 3D render of anomalies identified in GPR dataset. 

    Parameters: 
    dirPath (string): Current working directory filepath 
    XYZ (array): Cartesian coords. 
    A (array): Amplitude array 
    I (array): Isovalue bin values
    V (array): Isovalue bin assignment indexes
    Ci (array): Cluster assignment index (with assigned isovalue)
    Cc (array): Sample point in cluster with this centroid XYZ coord.    

    Returns:
    None

    """

    update('generating 3D plot...')

    # Initialise plot
    p = initialisePlot3D()

    # Recover no. isovalue bins
    nI = len(I)

    # ------------------

    # Storage 
    CentroidPoints = [] 
    CentrePoints = []
    MidPoints = []
    
    # Cycle isovalue bin indexes
    for i in range(nI):

        # extract sample point sets in ith isovalue bin
        mask = (V == i)    
        C_i = Ci[mask]                                       

        # extract unique clusters 
        # recall: [-1] --> no clusters identified
        C_j = np.unique(C_i)                                
        C_j = C_j[C_j >= 0]                                 
        nC = len(C_j)                                                                 

        # plot only if clusters are present...
        if (np.any(C_j >= 0)):

            # cycle unique cluster indexes [0,1,...]
            # note: ignore any 'unassigned' sample points 
            # (index = -1)
            for j in range(nC):

                # extract sample points in jth cluster 
                # (of the ith isovalue bin)
                mask = ((V == i) & (Ci == j))
                S = XYZ[mask,:]

                # extract centroid point coords. of this cluster
                CentroidPoint = Cc[mask][0]

                # swap y and z axis (s.t. y is vertical)
                S[:,[1,2]] = S[:,[2,1]]
                CentroidPoint[[1,2]] = CentroidPoint[[2,1]]

                # store centroid coordinate
                CentroidPoints.append(CentroidPoint)

                # wrap convex hull around point set
                # note: try statement in case cluster forms 
                # shape <3D (e.g. plane)
                try:
                    Mij = getMesh(S)
                except:
                    pass

                # compute centre point coordinate of mesh 
                # bounding box
                bbox = Mij.bounds
                CentrePoint = [(bbox[0] + bbox[1]) / 2, \
                               (bbox[2] + bbox[3]) / 2, \
                                (bbox[4] + bbox[5]) / 2]
                CentrePoints.append(CentrePoint)

                # compute midpoint of centroid and centre points
                MidPoint = (CentroidPoint + CentrePoint) / 2.0
                MidPoints.append(MidPoint)

                # append convex hull mesh to 3D plot
                p = addPlot3D_addMesh(p,Mij)

        # update console
        progress(nI,i,'...')
    
    # Convert store coords. to arrays 
    CentroidPoints = np.array(CentroidPoints)
    CentrePoints = np.array(CentrePoints)
    MidPoints = np.array(MidPoints)

    # Print to console (disable hidden elements)
    if True:
        update('return:')
        update('--> centroid point coords.')
        print('')
        with np.printoptions(threshold=np.inf):
            print(CentroidPoints)
        print('')
        update('--> centre point coords.')
        with np.printoptions(threshold=np.inf):
            print(CentrePoints)
        print('')
        update('--> mid point coords.')
        with np.printoptions(threshold=np.inf):
            print(MidPoints)
        print('')

    # Export to .txt
    if True:
        update('--> export to .txt ...')
        filePath = os.path.join(dirPath,'output_centroids.txt')
        with open(filePath,'w') as file:
            np.savetxt(filePath,CentroidPoints)
        filePath = os.path.join(dirPath,'output_centres.txt')
        with open(filePath,'w') as file:
            np.savetxt(filePath,CentrePoints)
        filePath = os.path.join(dirPath,'output_midpoints.txt')
        with open(filePath,'w') as file:
            np.savetxt(filePath,MidPoints)

    # Add lines between centres and centroids
    if True:
        N = np.shape(CentroidPoints)[0]
        for i in range(N):
            line = pv.Line(CentroidPoints[i,:], CentrePoints[i,:])
            p.add_mesh(line, color='k', line_width=1)

    # Add points at centres and centroids to plot 
    if True:
        p.add_points(CentroidPoints, color='c', point_size=5)
        p.add_points(CentrePoints, color='y', point_size=5)
        p.add_points(MidPoints, color='m', point_size=5)

    # ------------------

    # (~) Define RoI boundary outline - gprmax
    # note: domains always start from origin
    if True:
        if (mode == 'gprmax'):
            RoI_x = 0.42021
            RoI_y = 0.45012 
            RoI_z = 0.4199

    # (~) Define RoI boundary outline - vna/zond
    # note: domains always start from origin
    if True:
        if ((mode == 'vna') or (mode == 'zond')):
            RoI_x = 0.66
            RoI_y = 0.522 
            RoI_z = 0.53

    # (~) Define RoI boundary outline - pitscan
    # note: domains always start from origin 
    if True:
        if (mode == 'pitscan'):
            RoI_x = 10.00
            RoI_y = 1.20
            RoI_z = 1.435

    # Add RoI boundary outline 
    if True:
        cube = pv.Cube(center=(RoI_x/2, RoI_z/2, -RoI_y/2), 
                       x_length=RoI_x, 
                       y_length=RoI_z, 
                       z_length=RoI_y)
        p.add_mesh(cube, 
                   color='k', 
                   style='wireframe',
                   line_width=1.5)
        del cube

    # Add origin marker
    p = addPlot3D_origin(p)

    # Set view 
    p = set_camera(p)

    # ------------------

    # Export image files to working directory
    dirPath = os.path.dirname(os.path.abspath(__file__))
    p.save_graphic(os.path.join(dirPath,'output.svg'))    
    p.screenshot(os.path.join(dirPath,'output.png'))    
    p.save_graphic(os.path.join(dirPath,'output.pdf'))  

    # Display plot
    p.show()

    # ------------------
    return []

# Function to Generate 3D Meshes
def getMesh(S):
    """
    Converts sample point set into 3D convex hull via 
    QuickHull Method. 

    Parameters: 
    S (array): Nx3 sample point set 

    Returns:
    M (object): Equivalent convex hull (mesh) object.

    """

    # Compute closed convex hull
    H = ConvexHull(S)

    # Extract hull points (P) and faces (F)
    P = np.array(H.points)
    F = np.array(H.simplices)
    del H

    # Pad face array first column with no. points per face
    NF0,NF1 = np.shape(F)
    pad = np.full((NF0,1),NF1)
    F = np.concatenate((pad,F),axis=1)
    del pad
    
    # Convert to 1D array
    F = F.flatten()

    # Convert <ConvexHull> numpy array to point 
    # cloud object (i.e. 3D mesh) 
    M = pv.PolyData(P,F)

    # ------------------
    return M

# Function to Initialise 3D Visual Output
def initialisePlot3D():
    """
    Initialises scene.  

    Parameters: 
    None

    Returns:
    p (object): New plotter object.  

    """
    
    # Configure plotter object 
    p = pv.Plotter(off_screen=False)

    # Disable perspective
    p.parallel_projection = True

    # Create a Tkinter root window to get screen dimensions
    root = tk.Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()

    # Set the window size to the screen dimensions
    p.window_size = [screen_width, screen_height]

    # Set white background
    p.set_background('white')

    # Plot ascetics
    p = addPlot3D_origin(p)
    p = addPlot3D_orienter(p)
    p = addPlot3D_axes(p)

    # ------------------
    return p

# Function to Add 3D Origin Marker
def addPlot3D_origin(p):
    """
    Add origin marker to the scene. 

    Parameters: 
    p (object): Plotter object

    Returns:
    p (object): Updated

    """

    # Parameters
    lw = 4          # set line width                   
    c = 'grey'      # set line colour

    # ------------------

    # Show maker, +ve axes only
    if True:
        axL = np.array([[0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 1]],dtype=float) 
                                         
    # Show marker, +ve and -ve axes
    if False: 
        axL = np.array([[0, 0, 0],
                        [-1, 0, 0],
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 0, 0],
                        [0, -1, 0],
                        [0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0],
                        [0, 0, -1],
                        [0, 0, 0],
                        [0, 0, 1]],dtype=float)

    # ------------------

    # Scale axis lines length 
    #sx0 = p.bounds[0]/4; sx1 = p.bounds[1]/4
    #sy0 = p.bounds[2]/4; sy1 = p.bounds[3]/4
    #sz0 = p.bounds[3]/4; sz1 = p.bounds[5]/4
    
    # Fixed axis line length
    sx0 = 0.1; sx1 = 0.1
    sy0 = 0.1; sy1 = 0.1
    sz0 = 0.1; sz1 = 0.1 

    # Set axis length
    axL[1,0] = sx0*axL[1,0]
    axL[3,0] = sx1*axL[3,0]
    axL[5,1] = sy0*axL[5,1]
    axL[7,1] = sy1*axL[7,1]
    axL[9,2] = sz0*axL[9,2]
    axL[11,2] = sz1*axL[11,2]                       

    # Enable orientation rgb 
    if True:
        p.add_lines(axL[0:2,:], color=c, width=lw)
        p.add_lines(axL[2:4,:], color='r', width=lw)
        p.add_lines(axL[4:6,:], color=c, width=lw)
        p.add_lines(axL[6:8,:], color='g', width=lw)
        p.add_lines(axL[8:10,:], color=c, width=lw)
        p.add_lines(axL[10:12,:], color='b', width=lw)
    else: 
        p.add_lines(axL, color=c, width=lw)

    # Add black dot local coordinate origin marker
    if False:
        Mp = pv.Sphere(radius=0.075)
        p.add_mesh(Mp, color='black')
    
    # ------------------
    return p

# Function to Add 3D RGB Orientation Marker
def addPlot3D_orienter(p):
    """
    Adds RGB orientation marker in lower corner of scene. 

    Parameters: 
    p (object): Plotter object 

    Returns:
    p (object): Updated 

    """
    
    # Define orienter labels
    labels = ['x','z','y']

    # Add orienter icon
    orienter = dict(xlabel=labels[0], 
                    ylabel=labels[1], 
                    zlabel=labels[2])        
    p.add_axes(**orienter) 

    # ------------------
    return p

# Function to Add 3D Grid Lines 
def addPlot3D_grid(p):
    """
    Overlays grid markers on scene. 

    Parameters: 
    p (object): Plotter object 

    Returns:
    p (object): Updated

    """

    # Set grid style
    u_fontSize = 10
    u_fontStyle = 'times'
    u_fontColor = 'k'
    u_n_xticks = 6
    u_n_yticks = 6
    u_n_zticks = 6

    # Update grid
    p.show_grid(bounds=p.bounds,
                show_xaxis = True,
                show_yaxis = True,
                show_zaxis = True,
                show_xlabels = True,
                show_ylabels = True,
                show_zlabels = True,
                bold = False,
                font_size = u_fontSize,
                font_family = u_fontStyle,
                color = u_fontColor,
                xtitle= '', #'x [m]',
                ytitle= '', #'z [m]',
                ztitle= '', #'y [m]',
                n_xlabels = u_n_xticks,
                n_ylabels = u_n_yticks,
                n_zlabels = u_n_zticks,
                location = 'origin',
                ticks = 'inside',
                padding = 0,
                corner_factor = 0.5,
                minor_ticks = False,
                use_3d_text = False)

    # ------------------
    return p

# Function to Add 3D Axis Labels
def addPlot3D_axes(p):
    """
    Add axes to the scene.

    Parameters: 
    p (object): Plotter object 

    Returns:
    p (object): Updated  

    """

    # Set axes style
    u_fontSize = 12
    u_fontStyle = 'times'
    u_fontColor = 'k'
    u_n_xticks = 5
    u_n_yticks = 5
    u_n_zticks = 4

    # Update axes
    show = False
    p.show_bounds(bounds=p.bounds,
                    show_xaxis = show,
                    show_yaxis = show,
                    show_zaxis = show,
                    show_xlabels = show,
                    show_ylabels = show,
                    show_zlabels = show,
                    bold = True,
                    font_size = u_fontSize,
                    font_family = u_fontStyle,
                    color = u_fontColor,
                    xtitle= '', #'x [m]',
                    ytitle= '', #'z [m]',
                    ztitle= '', #'y [m]',
                    n_xlabels = u_n_xticks,
                    n_ylabels = u_n_yticks,
                    n_zlabels = u_n_zticks,
                    location = 'origin',
                    ticks = 'inside',
                    padding = 0,
                    corner_factor = 0.5,
                    minor_ticks = False,
                    use_3d_text = False)
    
    # ------------------
    return p

# Function to Append New 3D Mesh
def addPlot3D_addMesh(p,M):
    """
    Adds mesh to 3D scene.  

    Parameters: 
    p (object): Plotter object
    M (object): Mesh to add 

    Returns:
    p (object): Updated 

    """

    p.add_mesh(M,color='lightgrey',
               opacity=0.5,
               show_edges=True, 
               edge_color='lightgrey')

    # ------------------
    return p

# Function to Highlight 3D Visual Output Bounding Box
def addPlot3D_outline(p,M):
    """
    Adds outline encompassing collective meshes in scene. 

    Parameters: 
    p (object): Plotter object
    M (object): Combined meshes

    Returns:
    p (object): Updated   

    """

    # Set outline type - 'f' (full box), 'c' (corners only)
    outlineType = 'f'

    # Add outline
    if (outlineType == 'f'):
        p.add_mesh(M.outline(), 
                   color = 'k', 
                   line_width = 2, 
                   opacity = 1)   
    elif (outlineType == 'c'):
        p.add_mesh(M.outline_corners(), 
                   color = 'k', 
                   line_width = 2, 
                   opacity = 1)   
    else:
        raise Exception(
            f'invalid plot3D outline type: {outlineType}')

    # ------------------
    return p

# Function to Configure View
def set_camera(p):
    """
    Sets camera view for rendered scene. 

    Parameters: 
    p (object): Plotter object 

    Returns:
    p (object): Updated

    """

    # Uncomment As Required
    p.camera_position = 'iso'
    # p.camera.position = [1,1,1]
    # p.camera.focal_point = [0,0,0]
    # p.camera.view_up = [0,0,0]
    # p.camera.azimuth = -45
    # p.camera.elevation = -5
    # p.camera.roll = -95
    p.camera.zoom(0.95) 

    # ------------------
    return p
