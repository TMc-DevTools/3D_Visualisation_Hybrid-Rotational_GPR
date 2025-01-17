# Automated gprMax Area Scan Functionality
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
from scipy.fftpack import fftn, ifftn, fftshift, ifftshift
from scipy.constants import speed_of_light as c0
from sklearn.datasets import make_blobs

# ---------------------------------------------------------
# SETUP | HOUSEKEEPING
# ---------------------------------------------------------

# Function to Configure Log File
def clearLog():
    """
    Purges any existing log file.

    Parameters: 
    None

    Returns:
    None

    """ 

    # Purge any existing log file 
    dirPath = os.path.dirname(os.path.abspath(__file__))
    filePath = os.path.join(dirPath,'log.txt')
    if (os.path.exists(filePath)):
        os.remove(filePath)

    # -------------------
    return[]

# Function to Display General Update
def update(m):
    """
    Print timestamped update to console and to log file. 

    Parameters: 
    m (string): Message to relay

    Returns:
    None

    """   

    # Check current time
    t = datetime.now()

    # Format message
    msg = f'{t} | {m}'

    # Display to console
    print(msg)

    # Append to log file 
    dirPath = os.path.dirname(os.path.abspath(__file__))
    filePath = os.path.join(dirPath,'log.txt')
    with open(filePath,'a+') as file: 
        file.write(f'{msg}\n')

    # -------------------
    return[]

# Function to Display Progress Update
def progress(n,i,m):
    """
    Relay progress of current operation. 

    Parameters: 
    n (integer): Total no. operations to perform
    i (integer): Current operation index
    m (string): Descriptive message to display

    Returns:
    None

    """      

    # Set update frequency (i.e. print every % complete)
    p_show = 10 

    # Compute progress
    p_now = 100*(i/n) 
    p_last = 100*((i-1)/n)
    a = np.floor(p_now % p_show)
    b = np.floor(p_last % p_show)  
    
    # -------------------

    # Print update if required...
    if (i == 0) or (i == n-1) or (a < b):
        t = datetime.now()
        if (a < b):
            p_now = round(p_now / p_show) * p_show
            msg = f'{p_now:03.0f}%'
        elif(i == n-1):
            msg = f'{100:03.0f}%'
        else:
            msg = f'{0:03.0f}%'
        M = f'{msg} | {m}'
        update(M)
    
    # -------------------
    return[]

# Function to Recover Filenames/FilePaths in a Directory
def getFiles(dirPath,extensions):
    """
    Returns names and absolute paths of all files in directory ending 
    in the given extension(s).

    Parameters: 
    dirPath (string): Specified directory - absolute path
    extensions (array): File extensions to check for, including 
    . prefix 

    Returns:
    filePaths (array): List of relevant file paths - absolute
    fileNames (array): List of relevant file names 

    """  

    # Initialise storage arrays
    filePaths = []
    fileNames = []

    # Cycle extensions... 
    for e in extensions:

        # define filePath structure
        search_pattern = os.path.join(dirPath,f"*{e}")
        
        # locate matching files 
        filePaths_e = glob.glob(search_pattern)

        # extract filenames from filePaths
        fileNames_e = [os.path.basename(file) for \
                       file in filePaths_e]

        # append to storage 
        filePaths.extend(filePaths_e)
        fileNames.extend(fileNames_e)

    # Convert list to numpy array
    filePaths = np.array(filePaths)
    fileNames = np.array(fileNames)

    # ------------------------
    return filePaths, fileNames

# Function to Read Binary Files Byte by Byte
def readBytes(filePath, b0, dataType, nBytes):
    """
    Reads and returns specified bytes of binary file.   

    Parameters: 
    filePath (string): Specified binary file
    b0 (integer): Starting byte index - to read from [1]
    dataType (string): Type of data to expect to read 
                            'int' , 'float'
    nBtyes (integer): No. bytes to read [1]
    
    Returns:
    data (array): Encoded data - in base 10  

    """  

    # Configure binary read
    fmt = np.dtype(dataType)
    fmt.newbyteorder('<')

    # -------------------

    # Import bytes 
    with open(filePath,'rb') as file: 
        file.seek(b0)
        dataBytes = file.read(nBytes)

    # Extract human-readable value
    # note: if single element, returns non array object
    data = np.frombuffer(dataBytes,dtype=fmt)
    if (len(data) == 1):
        data = data[0]
        
    # -------------------
    return data

# Function to Create New Directory
def makeDir(dirPath):
    """
    Creates new directory.  

    Parameters: 
    dirPath (string): New directory - absolute path

    Returns:
    None

    """  
    
    # Remove any existing 
    if os.path.exists(dirPath):
        shutil.rmtree(dirPath)

    # Create new directory 
    os.makedirs(dirPath)

    # -------------------
    return []

# Function to Output Data to HDF5 File (Checkpointing)
def h5Out(filePath,data,handle):
    """
    Exports data to HDF5 file. 

    Parameters: 
    filePath (string): Specific input file
    data (array): Data to export
    handle (string): Dataset identifier

    Returns:
    None

    """  

    update('exporting data to h5 file (checkpointing)...')
    
    # -------------------

    # Ensure hdf5 file exists
    if not os.path.exists(filePath):
        with h5py.File(filePath,'w') as h5:
            pass

    # Open file for reading and writing
    with h5py.File(filePath,'r+') as h5:

        # remove existing dataset
        if handle in h5:
            del h5[handle]

        # save data to dataset <handle>
        D = h5.create_dataset(handle,data=data)  

    # -------------------
    return [] 

# Function to Import Data from HDF5 File (Checkpointing)
def h5In(filePath,handle):
    """
    Imports data from HDF5 file. 

    Parameters: 
    filePath (string): Specific input file
    handle (string): Dataset identifier

    Returns:
    data (array): Imported data

    """  

    update('importing data from h5 (checkpointing)...')

    # Check hdf5 file exists
    if os.path.exists(filePath):
        
        # open file for reading and writing
        with h5py.File(filePath,'r') as h5:

            # check dataset exists
            if handle in h5:
                data = h5[handle][:]
            else: 
                raise Exception(f'dataset not found: {handle}')
    else: 
        raise Exception(f'file not found: {filePath}')

    # -------------------
    return data

# ---------------------------------------------------------
# SUBROUTINES | TOOLS
# ---------------------------------------------------------

# Function to Add Box Geometry to Simulation 
def addBox(boxes,xyz0,xyz1,material='free_space',smoothing='n'):
    """
    Adds box to simulation. 

    Parameters: 
    boxes (object): Box storage object
    xyz0 (array): Vertex coords. - lower corner 
    xyz1 (array): Vertex coords. - upper corner 
    material (string): Geometry material handle
    smoothing (string): Toggle for boundary dielectric smoothing
                                        'n','y'

    Returns:
    boxes (object): Updated

    """  
    
    # Append new box object
    boxes.append([xyz0[0],xyz0[1],xyz0[2],
                  xyz1[0],xyz1[1],xyz1[2],
                  material,smoothing])
    
    # -------------------
    return boxes  

# Function to Add Cylinder Geometry to Simulation 
def addCylinder(cylinders,xyz0,xyz1,material='pec',smoothing='n'):
    """
    Adds box to simulation. 

    Parameters: 
    cylinders (object): Cylinder storage object
    xyz0 (array): Face centre coords. - front
    xyz1 (array): Face centre coords. - rear
    material (string): Geometry material handle
    smoothing (string): Toggle for boundary dielectric smoothing
                                        'n','y'

    Returns:
    cylinders (object): Updated

    """  

    # Add new cylinder object
    cylinders.append([xyz0[0],xyz0[1],xyz0[2],
                      xyz1[0],xyz1[1],xyz1[2],
                      material,smoothing])
    
    # -------------------
    return cylinders

# Function to Add Spherical Geometry to Simulation 
def addSphere(spheres,xyz0,r,material='pec',smoothing='n'):
    """
    Adds sphere to simulation. 

    Parameters: 
    spheres (object): Sphere storage object
    xyz0 (array): Centre coords. 
    material (string): Geometry material handle
    smoothing (string): Toggle for boundary dielectric smoothing
                                        'n','y'

    Returns:
    spheres (object): Updated

    """  

    # Add new sphere object
    spheres.append([xyz0[0],xyz0[1],xyz0[2],r,
                    material,smoothing])
    
    # -------------------
    return spheres 

# Function to Return Transect Parameters for Area Scan
def getPaths(nA,nB0,nB1,l,L,g,PML,INS,LFT):
    """
    Return area scan transect parameters. 

    Parameters: 
    nA (integer): No. traces 
    nB0 (integer): No transects (i.e. grid rows, along z axis) [1]
    nB1 (integer): No. transects (i.e. grid cols, along x axis) [1]  
    l (array): Side lengths - RoI [m]
    L (float): Side lengths - Global simulation domain [m] 
    g (float): Antenna absolute separation [m] 
    PML (float): Boundary thickness (absorbing) [m]
    INS (float): Geometry inset thickness [m]
    LFT (float): Antenna lift off [m]

    Returns:
    B (array): Transect parameters

    """  
    # Define storage
    B = []

    # Flag transect rows/cols [0/1]
    nB = nB0 + nB1
    rcB = np.zeros((nB,1))
    rcB[nB0:nB] = 1

    # Extract useful values
    lx, ly, lz = l
    Lx, Ly, Lz = L    

    # Define transect separation 
    dB0 = lz/(nB0-1)
    dB1 = lx/(nB1-1)
    dB = np.array([dB0,dB1])

    # ------------------------

    # Cycle transects
    for i in range(nB):
        
        # orientate antenna tx/rx
        if rcB[i] == 0: 
            rc = np.array([1,0,0])
        else:
            rc = np.array([0,0,1])

        # antenna separation vector
        g = rc*g

        # trace inc.
        dA = 1/(nA-1)
        dA = dA*l*rc

        # transect inc.
        if rcB[i] == 0:
            dB_i = np.array([0,0,dB[0]])
        else:
            dB_i = np.array([dB[1],0,0])

        # antenna midpoint home coord.
        x0 = PML + INS   
        y0 = PML + INS + ly + LFT
        z0 = PML + INS
        xyz0 = np.array([x0,y0,z0])

        # initial position (tx)
        if rcB[i] == 0:
            tx0 = xyz0 - (g/2) + i*dB_i
        else: 
            tx0 = xyz0 - (g/2) + (i-nB0)*dB_i

        # initial position (rx)
        rx0 = tx0 + g

        # append to storage
        B_now = np.hstack((tx0,rx0,dA))
        B.append(B_now)  

    # ------------------------
    return B

# Function to Return Transect Path Preview 
def makeScanTrajPreview(P,nB0,nB1,L,l,delta,g,B):
    """
    Return area scan transect parameters. 

    Parameters:
    P (array): Consolidated scan parameters
    nB0 (integer): No transects (i.e. grid rows, along z axis) [1]
    nB1 (integer): No. transects (i.e. grid cols, along x axis) [1] 
    L (float): Side lengths - Global simulation domain [m] 
    l (array): Side lengths - RoI [m]
    delta (array): Spatial separations [m]
    g (float): Antenna absolute separation [m] 
    B (array): Transect parameters

    Returns:
    None

    """  

    update(f'previewing scanning trajectories...')

    # Extract key parameters
    pfx = P[0]
    dirPath = P[2]
    
    # Flag area scan ROWS (0) and COLS (1)
    nB = nB0 + nB1
    rcB = np.zeros((nB,1))                     
    rcB[nB0:nB] = 1 

    # ------------------------

    # Create figure and axis                                   
    fig, ax = plt.subplots(figsize=(5,5))               
    ax.set_axisbelow(True)                             
    ax.set_title(f'{pfx}: Bscan Trajectories')           
    plt.xlabel('x [m]')                                 
    plt.ylabel('z [m]')   
    plt.gca().set_aspect('equal')                      

    # Maximise figure window
    #manager = plt.get_current_fig_manager()
    #manager.full_screen_toggle() 

    # Move figure to lower left corner of screen
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry(f"+{100}+{100}")

    # Plot boundaries
    Lx = L[0]
    Lz = L[2]
    ax = plotNestedRectangles(ax,Lx,Lz,delta)

    # Additional variables 
    arrow_coords = []                               

    # Define trajectory arrows 
    for i in range(nB):                             

        # start/end coords.
        tx_start_x = B[i][0]                      
        tx_start_z = B[i][2]                        
        rx_start_x = B[i][3]
        rx_start_z = B[i][5]
        if rcB[i] == 0:                            
            tx_end_x =  tx_start_x + g/2 + \
                (l[0] + delta[1])-g/2 
            rx_end_x =  tx_end_x + g 
            tx_end_z = tx_start_z                                   
            rx_end_z = rx_start_z                                                       
        elif rcB[i] == 1:                           
            tx_end_x = tx_start_x
            tx_end_z = tx_start_z + g/2 + (l[2] + delta[1])-g/2 
            rx_end_x = rx_start_x                                                       
            rx_end_z = tx_end_z + g                                    
        arrow_coords.append([(tx_start_x,tx_start_z),
                             (rx_start_x,rx_start_z),
                             (tx_end_x,tx_end_z),
                             (rx_end_x,rx_end_z)])

    # Cleanup temporary variables
    del tx_start_x, tx_start_z, rx_end_x, rx_end_z  

    # Determine element for first row and first col
    index_first_row = np.where(rcB == 0)[0][0]
    index_first_col = np.where(rcB == 1)[0][0]

    # Plot trajectory arrows
    i = 0                                                                          
    for tx_start, rx_start, tx_end, rx_end in arrow_coords:
    
        # add an arrow with specified start and end points
        ax.annotate('', xy=rx_end, xytext=tx_start,
                    arrowprops=dict(arrowstyle='->', 
                                    color='k',
                                    zorder=3))
        
        # add Bscan index label
        ax = addBscanIndex(ax,i,rcB,L,tx_start,delta)

        # define circle marker size
        marker_size = 2.5e-3

        # add SOLID RED circle at <tx> START point 
        circle = plt.Circle(tx_start, 
                            marker_size, 
                            color='r',
                            zorder=5)
        circle.set_fill(True)
        ax.add_patch(circle)

        # add HOLLOW BLUE circle at <rx> END point 
        circle = plt.Circle(rx_end, 
                            marker_size, 
                            color='b',
                            zorder=5)
        circle.set_fill(False)
        ax.add_patch(circle)

        # for clarity, add these markers only on first row and col
        #if (i == index_first_row) or (i == index_first_col):

        # add SOLID BLUE circle at <rx> START point 
        circle = plt.Circle(rx_start, 
                            marker_size, 
                            color='b',
                            zorder=5)
        circle.set_fill(True)
        ax.add_patch(circle)

        # add HOLLOW RED circle at <tx> END point 
        circle = plt.Circle(tx_end, 
                            marker_size, 
                            color='r',
                            zorder=5)
        circle.set_fill(False)
        ax.add_patch(circle)   
        
        # update counter
        i += 1

    # Mark the RoI centre with a cross
    ax = addCentreMark(ax,L)

    # Set axis limits
    offset = 2*delta[0]
    plt.xlim(0, Lx + offset)
    plt.ylim(0, Lz + offset) 

    # Shade absorbing boundary region
    plt.fill_between([0,Lx], 0, delta[0], 
                     color='dimgrey', alpha=0.3)
    plt.fill_between([0,Lx], Lz-delta[0], Lz, 
                     color='dimgrey', alpha=0.3)
    plt.fill_between([0,delta[0]],delta[0],Lz-delta[0], 
                     color='dimgrey', alpha=0.3)
    plt.fill_between([Lx-delta[0],Lx],delta[0],Lz-delta[0], 
                     color='dimgrey', alpha=0.3)

    # Add grid 
    ax.grid(True, alpha = 0.3)  

    # ------------------------

    # Export figure
    filename = f'{pfx}_trajectories'
    filepath = os.path.join(dirPath,filename)
    extensions = ['pdf','svg','png']
    for ext in extensions:
        filepath_now = filepath + '.' + ext
        plt.savefig(filepath_now, format=ext, dpi=350)

    # ------------------------

    # Preview the plot with countdown 
    plt.show(block=False)                   
    duration = 5                                
    fig = plt.gcf()
    timer_text = fig.text(0.95, 0.95, f"Time left: {duration}", 
                          fontsize=6, ha='right')
    for i in range(duration, 0, -1):
        timer_text.set_text(f"Time left: {i}")
        plt.pause(1) 
    plt.close() 

# Function to Plot Transect Trajectories 
def plotNestedRectangles(ax,x,y,delta):
    """
    Plot nested rectangles for key boundaries. 

    Parameters:
    ax (object): Specific plot axis
    x (float): Vertex coord. - closest to global datum [m]
    y (float): = 
    delta (array): Spatial separations [m]

    Returns:
    ax (object): Updated

    """  

    # Add outermost rectangle (global simulation boundary)
    outer_rectangle = plt.Rectangle((0, 0), x, y, 
                                    edgecolor='black', 
                                    facecolor='none',
                                    linewidth=1)
    ax.add_patch(outer_rectangle)

    # Add middle rectangle (pml boundary --> inset = delta0)
    botLeft_corner = (delta[0], delta[0])
    x_length = x-(2*delta[0])
    z_length = y-(2*delta[0])
    new_rectangle = plt.Rectangle(botLeft_corner, 
                                  x_length, z_length, 
                                  edgecolor='k', linestyle ='--', 
                                  facecolor='none')
    ax.add_patch(new_rectangle)

    # Add inner rectangle 
    botLeft_corner = (delta[0] + delta[1],delta[0] + delta[1])
    x_length = x-2*(delta[0] + delta[1])
    z_length = y-2*(delta[0] + delta[1])
    new_rectangle = plt.Rectangle(botLeft_corner, 
                                  x_length, z_length, 
                                  edgecolor='orange', 
                                  linestyle ='--', 
                                  facecolor='none')
    ax.add_patch(new_rectangle)
    
    # ------------------------
    return ax

# Function to Overlay Transect Indexes
def addBscanIndex(ax,i,rcB,L,tx_start,delta):
    """
    Adds identifying index beside transects in preview. 

    Parameters:
    ax (object): Specific plot axis
    i (integer): Index of transect of interest [1] 
    rcB (array): Row/col index flags [1]
    L (float): Side lengths - Global simulation domain [m] 
    tx_start (array): Coords. of tx antenna start point [m]
    delta (array): Spatial separations [m]

    Returns:
    ax (object): Updated
    
    """   

    # Show only necessary indexes on plot
    first_one_index = next((i for i, val in enumerate(rcB) \
                            if val == 1), None)

    # Toggle showing every transect label
    if (True):
        allowed_indexes = [0,1,first_one_index-1,
                           first_one_index,len(rcB)-1]

    # Toggle showing every other transect label
    if (True):        
        allowed_indexes = [num for num in range(0, len(rcB)-1 + 1) \
                           if num % 2 == 0]      

    # Add to preview
    if (i in allowed_indexes):
        Lx = L[0]
        Lz = L[2]
        p = delta[0]               
        if (rcB[i] == 0):          
            x = Lx + p
            z = tx_start[1]
        elif (rcB[i] == 1):        
            x = tx_start[0]
            z = Lz + p
        ax.text(x, z, f'$B_{{{i}}}$', fontsize=8, ha='center', 
                va='center', color='k')

    # ------------------------
    return ax

# Function to Add Centre Marks
def addCentreMark(ax,L):
    """
    Adds cross to the centre point of the RoI footprint

    Parameters:
    ax (object): Specific plot axis
    L (float): Side lengths - Global simulation domain [m] 

    Returns:
    ax (object): Updated
    
    """  

    # Compute centre point
    x = L[0]/2
    z = L[2]/2
    scl = 5e-3
    a = 0.3

    # Add marker - intersecting diagonal lines
    ax.plot([x - scl, x + scl], [z - scl, z + scl], 
            color='black',alpha = a)  
    ax.plot([x - scl, x + scl], [z + scl, z - scl], 
            color='black',alpha = a) 

    # ------------------------
    return ax

# Function to Generate Windows Automation Files 
def makeBAT(P,filename,cmds):
    """
    Creates <.BAT> automation files for windows. 

    Parameters:
    P (array): Consolidated scan parameters
    filename (string): Name of file to generate
    cmds (array): Array of automation command to include

    Returns:
    None

    """  

    # Extract key parameters
    pfx = P[0]
    dirPath = P[2]

    # Create <.bat> file
    filepath = os.path.join(dirPath,filename)
    file_BAT = open(filepath,'w')

    # Add <.bat> commands
    for line in cmds:
        file_BAT.write(line)

    # Close <.bat> file
    file_BAT.close

    # Progress update
    update(f'success: created {filepath}')

# ---------------------------------------------------------
# SUBROUTINES | MAIN
# ---------------------------------------------------------

def areaScan_gprmax():
    """
    Generates automation scripts for gprMax area scan
    For how configure gprMax, see https://www.gprmax.com/   

    Parameters: 
    None - Defined in syntax below...

    Returns:
    None

    Notes:

    configure output directory
        (-) all simulation files saved in temporary directory 
            within GPR data repository
        (-) user must copy files to gprMax top-level directory in 
            order to run  

    
    general
        (-) datum is in bottom LHS corner of simulation domain
        (-) gprMax advises domain size = RoI + PML + Inset 
            RoI -----> subsurface region of interest 
            PML -----> 15x cells for PML boundary layer 
            Inset ---> target placement inset from PML layer 
        (-) gprMax demo uses 0m lift off  

    custom materials
        (-) new array row => new material 
        (-) row format: [rep,ec,rmp,ml,h]
                rep --> relative electrical permittivity [1]
                ec ---> electrical conductivity [siemens/m]
                rmp --> relative magnetic permeability [1]
                ml ---> magnetic loss [ohms/m]
                h --> reference handle
            
    commands to add targets
        (-) boxes = add_box(boxes,xyz0,xyz1,material,smoothing)   
        (-) cylinders = add_cylinder(
                            cylinders,xyz0,xyz1,r,material,smoothing)
        (-) spheres = add_sphere(spheres,xyz0,r,material,smoothing)

        (-) <xyz_> = coordinate array [m]
        (-) <r> = radius [m]

    transect trajectories
        (-) antenna midpoint always starts on the corner of the RoI 
            closest to global datum
        (-) antenna midpoint maintains set lift-off throughout all 
            transects 

    """  

    # Configure output directory 
    dirPath = r'...'
    dirPath_output = os.path.join(dirPath,'data','(-)')
    makeDir(dirPath_output)

    # Initialise storage 
    boxes = []
    cylinders = []
    spheres = []

    # Initialise datum
    datum = np.zeros(3)

    # ------------------------

    # Metadata - I
    ff = '.8e'                          # output precision [1]
    nA = 51                             # no. traces per transect [1]
    nB0 = 51                            # no. transect rows [1]
    nB1 = 51                            # no. transect cols [1]   

    # Metadata - II
    title = ''

    # EM-modelling
    dxdydz = np.array([2e-3,2e-3,2e-3])         # cell resolution [m]                                   
    t = 5e-9                                    # time domain [s]
    
    # Hardware
    fc = 1.5e9                          # centre frequency [Hz]
    f_scl = 1                           # amplitude scaling factor [1]    
    f_type = 'ricker'                   # waveform type [str]
    tx_pol = 'z'                        # transmitter (tx) polarity
    tx_handle = 'my_ricker'             # transmitter handle [str]
    g = 40e-3                           # tx/rx separation

    # Lift off
    LFT = 0                             # antenna lift off [m]

    # RoI
    lx = 0.50                           # sample side length [m]
    ly = 0.25                           # sample vertical depth [m]  
    lz = 0.50                           # sample side length [m]

    # Simulation domain
    PML = 15*np.max(dxdydz)             # compute 15x cell boundary
    INS = 15*np.max(dxdydz)             # compute 15x cell boundary
    Lx = lx + 2*PML + 2*INS             # domain horizontal span [m]
    Ly = ly + 2*PML + 2*INS + LFT       # domain vertical span [m]
    Lz = lz + 2*PML + 2*INS             # domain horizontal span [m]

    # Custom materials
    M = np.array([[4,0,1,0,'drySand_homo']]) 

    # Dependants
    nB = nB0 + nB1                      # total no. transects 
    l = np.array([lx,ly,lz])            # RoI span
    L = np.array([Lx,Ly,Lz])            # simulation domain span 
    Cx = Lx/2                           # central coords. of RoI  
    Cy = PML + INS + ly/2               
    Cz = Lz/2                                                      

    # Useful geometry values
    x1 = 274e-3
    x2 = x1 + 43e-3
    y1 = PML + INS + ly - 130e-3
    z2 = 49.5e-3
    z3 = 69.5e-3

    # Define targets
    s1_r = 50e-3; s1_M = 'pec'; s1_xyz0 = np.array([x1, y1, Cz])
    s2_r = 50e-3; s2_M = 'pec'; s2_xyz0 = np.array([x2, y1, Cz+z2])
    s3_r = 50e-3; s3_M = 'pec'; s3_xyz0 = np.array([x2, y1, Cz-z3]) 

    # Add targets
    spheres = addSphere(spheres, s1_xyz0, s1_r, s1_M)
    spheres = addSphere(spheres, s2_xyz0, s2_r, s2_M)
    spheres = addSphere(spheres, s3_xyz0, s3_r, s3_M)

    # Add RoI
    boxes = addBox(boxes,datum,L,M[0,4])

    ## Geometry feedback | setup
    #G_xyz0 = datum                 # lower LHS corner 
    #G_xyz1 = L                     # upper RHS corner 
    #G_dxdydz = dxdydz              # spatial resolution 
    #G_fine = 'n'                   # fine output toggle

    # Geometry feedback | handles
    #G_handles = []                                
    #for i in range(nB):
    #    G_handles.append(f'b{i:0>3}_a')

    # ------------------------
    
    # Generate scan paths
    B = getPaths(nA,nB0,nB1,l,L,g,PML,INS,LFT)

    # ------------------------

    # Save copy of this script to output directory
    fileName = 'tools.py'
    filePath0 = os.path.abspath(__file__)
    filePath1 = os.path.join(dirPath_output,fileName)
    shutil.copy2(filePath0, filePath1)

    # ------------------------

    # Define <.IN> file commands
    cmds = []
    cmds.append(f'#title: {title}')                                                                       
    cmds.append(f'#domain: {Lx:{ff}} {Ly:{ff}} {Lz:{ff}}')               
    cmds.append(f'#dx_dy_dz: {dxdydz[0]:{ff}} {dxdydz[1]:{ff}} 
                {dxdydz[2]:{ff}}')                                           
    cmds.append(f'#time_window: {t:.1e}')
    cmds.append(f'#waveform: {f_type} {f_scl} {fc:.1e} {tx_handle}')   
    n = np.shape(M)[0]
    for i in range(n):
        cmds.append(f'#material: {M[i,0]} {M[i,1]} {M[i,2]} {M[i,3]} 
                    {M[i,4]}')   
    for bx in boxes:
        cmds.append(f'#box: {bx[0]:{ff}} {bx[1]:{ff}} {bx[2]:{ff}} 
                    {bx[3]:{ff}} {bx[4]:{ff}} {bx[5]:{ff}} {bx[6]} 
                    {bx[7]}')
    for cy in cylinders:
        cmds.append(f'#cylinder: {cy[0]:{ff}} {cy[1]:{ff}} 
                    {cy[2]:{ff}} {cy[3]:{ff}} {cy[4]:{ff}} 
                    {cy[5]:{ff}} {cy[6]:{ff}} {cy[7]} {cy[8]}')
    for sp in spheres:
        cmds.append(f'#sphere: {sp[0]:{ff}} {sp[1]:{ff}} {sp[2]:{ff}} 
                    {sp[3]:{ff}} {sp[4]} {sp[5]}')   
    
    # Create <.IN> files 
    for i in range(nB):
        
        # recover global parameters (same for all transects)
        cmds_B = cmds[:]

        # define local variables (vary transect by transect)
        cmds_B.append(f'#hertzian_dipole: {tx_pol} {B[i][0]:{ff}} 
                      {B[i][1]:{ff}} {B[i][2]:{ff}} {tx_handle}')
        cmds_B.append(f'#rx: {B[i][3]:{ff}} {B[i][4]:{ff}} 
                      {B[i][5]:{ff}}')
        cmds_B.append(f'#src_steps: {B[i][6]:{ff}} {B[i][7]:{ff}} 
                      {B[i][8]:{ff}}')
        cmds_B.append(f'#rx_steps: {B[i][6]:{ff}} {B[i][7]:{ff}} 
                      {B[i][8]:{ff}}')
        #cmds_B.append(f'#geometry_view: {G_xyz0[0]:{ff}} 
        # {G_xyz0[1]:{ff}} {G_xyz0[2]:{ff}} {G_xyz1[0]:{ff}} 
        # {G_xyz1[1]:{ff}} {G_xyz1[2]:{ff}} {G_dxdydz[0]:{ff}} 
        # {G_dxdydz[1]:{ff}} {G_dxdydz[2]:{ff}} {G_handles[i]} 
        # {G_fine}')
        
        # export to <.IN> file
        fileName = f'b{i:0>3}_a.in'
        filePath = os.path.join(dirPath_output,fileName)
        try: 
            with open(filePath,'w') as file:
                for line in cmds_B:
                    file.write(f'{line}\n')
        except:
            raise Exception(f'error exporting: {filePath}')

    # ------------------------

    # Preview scan paths
    # /// see <makeScanTrajPreview()> above ///

    # Generate automation scripts | <.bat> (Windows)
    # /// See <makeBAT()> above ///

    # ------------------------

    # Generate automation scripts | <.sh> (Linux)

    # Define <.sh> commands
    cmds = []
    for i in range(nB):
        cmds.append(
            f'python -m gprMax b{i:0>3}_a.in -n {nA}\n')
        cmds.append(
            f'python -m tools.outputfiles_merge b{i:0>3}_a\n')
    
    # Export to <.sh> file
    filePath = os.path.join(dirPath_output,'run.sh')
    with open(filePath, 'w') as file:
        for line in cmds:
            file.write(line)
            
    # ------------------------
    return []
