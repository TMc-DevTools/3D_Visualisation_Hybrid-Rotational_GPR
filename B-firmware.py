# Laboratory Hybrid-Rotational GPR Firmware
import os
import shutil
import time
import numpy as np
import datetime
import RPi.GPIO as GPIO
import contextlib
import pyvisa as visa
with contextlib.redirect_stdout(None):
    import pygame

# ---------------------------------------------------------
# SETUP | GENERAL
# ---------------------------------------------------------

# VNA Static IP
vnaIP = '192.168.113.206'

# Limit Switches
limitSwitch1_xMin = 5
limitSwitch2_xMax = 6
limitSwitch3_tMin = 16
limitSwitch4_tMax = 26

# Motor | Azimuth (t)
t_direction_pin = 20
t_step_pin = 21

# Motor | Lateral (x)
x_direction_pin = 24
x_step_pin = 23
x_ms1_pin = 17
x_ms2_pin = 27
x_enable_pin = 22

# Configuration | GPIO Names
GPIO.setmode( GPIO.BCM )

# Configuration | Limit Switches
GPIO.setup(limitSwitch1_xMin,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.setup(limitSwitch2_xMax,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.setup(limitSwitch3_tMin,GPIO.IN,pull_up_down=GPIO.PUD_UP)
GPIO.setup(limitSwitch4_tMax,GPIO.IN,pull_up_down=GPIO.PUD_UP)

# Configuration | Motor (t)
GPIO.setup(t_direction_pin, GPIO.OUT)
GPIO.setup(t_step_pin, GPIO.OUT)

# Configuration | Motor (x)
GPIO.setup(x_direction_pin, GPIO.OUT)
GPIO.setup(x_step_pin, GPIO.OUT)
GPIO.setup(x_ms1_pin, GPIO.OUT)
GPIO.setup(x_ms2_pin, GPIO.OUT)
GPIO.setup(x_enable_pin, GPIO.OUT)

# Function to Initialise Motors 
def rst():
    """
    Resets motor GPIO pin states.

    Parameters: 
    None 

    Returns: 
    None

    """

    # Reset t-motor pin states
    # - 

    # Reset x-motor pin states
    GPIO.output(x_direction_pin, GPIO.LOW)
    GPIO.output(x_step_pin, GPIO.LOW)
    GPIO.output(x_ms1_pin, GPIO.LOW)
    GPIO.output(x_ms2_pin, GPIO.LOW)
    GPIO.output(x_enable_pin, GPIO.LOW)
rst()

# ---------------------------------------------------------
# SETUP | MOVEMENT 
# ---------------------------------------------------------

# Configuration | Homing Retraction
nRetract_x = 50
nRetract_t = 1000

# Configuration | Speed of Motion
# Motor (x) | Fast = 07 | Slow = 0.5
# Motor (t) | Fast = 10 | Slow = 5.0
vx_slow = 0.1
vx_fast = 10.0
vx_vfast = 50.0
vt_slow = 5
vt_fast = 10
vt_vfast = 20

# Configuration | Steps Per Full Motor Revolution 
# note: At Max Microstepping Setting
x_DIP = 3200
t_DIP = 40e3

# Hardware | Motor (x) GT3 Pulley
x_pulley_pitch = 3e-3
x_pulley_teeth = 20

# Conversion Factors
# lx2nx --> nx = lx/alpha | minimal x movement (1 step)
# lt2nt --> nt = lt/beta | minimal t movement (1 step)
alpha = ((x_pulley_teeth)*(x_pulley_pitch))/x_DIP   
beta = 360/t_DIP                                    

# Buffer | Wait Time 
t_BUFF = 0.5

# ---------------------------------------------------------
# SUBROUTINES | MAIN 
# ---------------------------------------------------------

# Function to Advance Motor (x) by Single Microstep
def step_x(dir,v):
    """
    Advances Motor (x) by single step. 

    Parameters: 
    dir (integer): move forwards (1) or backwards (0) [1]
    v (float): Movement Speed [1]

    Returns:
    None 

    """

    # Set pulse delay (i.e. advance speed)
    t = 0.001/v

    # Set direction
    if (dir == 1):
        GPIO.output(x_direction_pin, GPIO.LOW)    # LOW = forward
    elif (dir == 0):
        GPIO.output(x_direction_pin, GPIO.HIGH)
    else:
        raise ValueError(f'invalid stepping direction')

    # Enable microstepping
    GPIO.output(x_ms1_pin,GPIO.HIGH)
    GPIO.output(x_ms2_pin,GPIO.HIGH)

    '''
     no microstepping ...... MS1 = LOW | MS2 = LOW
     1/2 stepping .......... MS1 = HIGH | MS2 = LOW
     1/4 stepping .......... MS1 = LOW | MS2 = HIGH
     1/8 stepping .......... MS1 = HIGH | MS2 = HIGH
    '''

    # Movement cycle (single step)
    GPIO.output(x_step_pin,GPIO.HIGH)
    time.sleep(t)
    GPIO.output(x_step_pin,GPIO.LOW)
    time.sleep(t)

# Function to Advance Motor (x) by Multiple Microsteps 
def advance_x(N,v):
    """
    Advances Motor (x) by N steps at speed v. 

    Parameters: 
    N (integer): Number of steps to move [1]
    v (float): Movement speed [1] 

    Returns:
    None 

    """

    # Set direction
    if (N > 0):
        dir = 1
        h = 'xmax'
    elif (N < 0):
        dir = 0
        h = 'xmin'
    else:
        raise ValueError('(!): invalid step request (x-motor)')

    # Set steps remaining
    i = abs(N)

    # Movement cycle
    time.sleep(t_BUFF)
    while (i > 0):
        check4danger(h)
        step_x(dir,v)
        i -= 1

    # Buffer
    time.sleep(t_BUFF)

# Function to Home Motor (x)
def home_x():
    """
    Homes motor (x).

    Parameters: 
    None

    Returns:
    None 

    """

    # Home (fast)
    i = 0
    while (switchRead(limitSwitch1_xMin) != 0):
        step_x(0,vx_vfast)
        if (i > nRetract_x):
            check4danger('xmax')
        i += 1
    time.sleep(t_BUFF)

    # Retract (slow)
    i = nRetract_x
    while (i > 0):
        step_x(1,vx_slow)
        check4danger('xmax')
        i -= 1
    time.sleep(t_BUFF)

    # Home (slow)
    while (switchRead(limitSwitch1_xMin) != 0):
        step_x(0,vx_slow)
        check4danger('xmax')
    time.sleep(t_BUFF)

# Function to Max-Out Motor (x)
def max_x():
    """
    Homes motor (x), max displacement and prints total step count.  

    Parameters: 
    None

    Returns:
    None 

    """

    # Update console
    update('maxing out t ...')

    # Home x-motor
    home_x()

    # Max out (fast)
    i = 0
    while (switchRead(limitSwitch2_xMax) != 0):
        step_x(1,vx_fast,mu=False)
        if (i > nRetract_x):
            check4danger('xmin')
        i += 1
    time.sleep(t_BUFF)

    # Retract (slow)
    j = nRetract_x
    while (j > 0):
        step_x(0,vx_slow,mu=False)
        check4danger('xmin')
        i -= 1
        j -= 1
    time.sleep(t_BUFF)

    # Max out (slow)
    while(switchRead(limitSwitch2_xMax) != 0 ):
        step_x(1,vx_slow,mu=False)
        check4danger('xmin')
        i += 1
    time.sleep(t_BUFF)

    # Predicted values
    Lx0 = 1.505
    xC = x_pulley_pitch * x_pulley_teeth
    dx0 = xC/(x_DIP-1)
    Nx0 = (Lx0/dx0)+1

    # Measured values
    Nx1 = i
    Lx1 = Lx0
    dx1 = Lx0/(i-1)

    # Return
    update(f'--> <max_x> ....... = {Nx1} [step]')
    update(f'--> <predicted> ... = {Nx0} [step]')
    update(' ')
    update(f'--> <max_x> ....... = {Lx1} [m]')
    update(f'--> <measured> .... = - [m]')
    update (' ')
    update(f'--> <dx> .......... = {dx1} [m]')
    update(f'--> <predicted> ... = {dx0} [m]')

# Function to Indefinitely Bounce Motor (x) Between Extremal Limits 
def bounce_x():
    """
    Moves system back and forth between extremal lateral 
    displacements indefinitely. 

    Parameters: 
    None

    Returns:
    None 

    """

    # Update console
    update('bouncing x ...')

    # Home x-motor
    home_x()

    # Prime x-motor
    advance_x(2000,vx_fast)

    # Bounce loop
    time.sleep(2)
    while True:
        advance_x(2000,vx_fast)
        time.sleep(1)
        advance_x(-2000,vx_fast)
        time.sleep(1)

# Function to Trigger GPR Trace Measurement
def vnaScanNow(vnaIP,id,i,j):
    """
    Triggers VNA linear frequency sweep using currently set VNA 
    parameters.   

    Parameters: 
    vnaIP (string): Static VNA IP address
    id (integer): Unique trace ID [1]
    i (integer): Current trace index in transect [1]
    j (integer): Current transect index in full scanning pass [1]

    Returns:
    None 

    """
    buffer = 0.1
    try:
        # Open connection to the VNA
        rm = visa.ResourceManager()
        vna = rm.open_resource(f"TCPIP::{vnaIP}::INSTR")

        # Set save location to onboard SD card
        vna.write(':MMEMory:CDIRectory "%s"' % ('[SDCARD]:'))

        # Trigger the measurement
        vna.write("INIT:IMM")

        # Wait for the measurement to complete
        vna.query("*OPC?")

        # Retrieve measurement data
        vna.query("CALC:DATA:FDAT?")

        # Write output to SD card
        filetype = '.s1p'
        filename = f'b{i:0>6}_a{j:0>6}'
        vna.write(f':MMEMory:STORe:SNP:DATA "{filename}{filetype}"')
    except visa.VisaIOError as e:
        update("(!) Error:", e)
    finally:
        # Close the connection to the VNA
        vna.close()
        rm.close()

        # Buffer
        time.sleep(buffer)

# Function to Advance Motor (t) by Single Microstep
def step_t(dir,v):
    """
    Advances Motor (t) by single step. 

    Parameters: 
    dir (integer): move forwards (1) or backwards (0) [1]
    v (float): Movement Speed [1]

    Returns:
    None 

    """

    # Set pulse delay (i.e. rotation speed)
    s = (1e-3)/v

    # Direction select
    if (dir == 0):
        GPIO.output(t_direction_pin,1)
    elif (dir == 1):
        GPIO.output(t_direction_pin,0)
    else:
        raise ValueError(
            '(!) error: invalid direction request (t-motor)')

    # Move single step
    GPIO.output(t_step_pin,GPIO.HIGH)
    time.sleep(s)
    GPIO.output(t_step_pin,GPIO.LOW)
    time.sleep(s/2)

# Function to Advance Motor (t) by Multiple Microsteps 
def advance_t(N,v):
    """
    Advances Motor (t) by N steps at speed v. 

    Parameters: 
    N (integer): Number of steps to move [1]
    v (float): Movement speed [1] 

    Returns:
    None 

    """

    # Set direction
    if (N > 0):
        dir = 1
        h = 'tmax'
    elif (N < 0):
        dir = 0
        h = 'tmin'
    else:
        raise ValueError('(!): invalid step request (t-motor)')

    # Set steps remaining
    i = abs(N)

    # Movement cycle
    time.sleep(t_BUFF)
    while (i > 0):
        check4danger(h)
        step_t(dir,v)
        i -= 1

    # Buffer
    time.sleep(t_BUFF)

# Function to Home Motor (t)
def home_t():
    """
    Homes motor (t).

    Parameters: 
    None

    Returns:
    None 

    """

    # Home (fast)
    i = 0
    while (switchRead(limitSwitch3_tMin) != 0):
        step_t(0,vt_vfast)
        if (i > nRetract_t):
            check4danger('tmax')
        i += 1
    time.sleep(t_BUFF)

    # Retract (slow)
    i = nRetract_t
    while (i > 0):
        step_t(1,vt_slow)
        check4danger('tmax')
        i -= 1
    time.sleep(t_BUFF)

    # Home (slow)
    while (switchRead(limitSwitch3_tMin) != 0):
        step_t(0,vt_slow)
        check4danger('tmax')
    time.sleep(t_BUFF)

# Function to Max-Out Motor (t)
def max_t():
    """
    Homes motor (t), max displacement and prints total step count.  

    Parameters: 
    None

    Returns:
    None 

    """

    # Update console
    update('maxing out t ...')

    # Home t-motor
    home_t()

    # Max out (fast)
    i = 0
    while (switchRead(limitSwitch4_tMax) != 0):
        step_t(1,vt_fast)
        if (i > nRetract_t):
            check4danger('tmin')
        i += 1
    time.sleep(t_BUFF)

    # Retract (slow)
    j = nRetract_t
    while (j > 0):
        step_t(0,vt_slow)
        check4danger('tmin')
        i -= 1
        j -= 1
    time.sleep(t_BUFF)

    # Max out (slow)
    while(switchRead(limitSwitch4_tMax) != 0 ):
        step_t(1,vt_slow)
        check4danger('tmin')
        i += 1
    time.sleep(t_BUFF)

    # Predicted values
    dt0 = 360/(t_DIP-1)
    Lt0 = 210
    Nt0 = (Lt0/dt0)+1

    # Measured values
    Nt1 = i
    Lt1 = 360*(i/(t_DIP-1))
    dt1 = Lt1/(i-1)

    # Return
    update(f'--> <max_t> ....... = {Nt1} [step]')
    update(f'--> <predicted> ... = {Nt0} [step]')
    update(' ')
    update(f'--> <max_t> ....... = {Lt1} [deg.]')
    update(f'--> <measured> .... = {Lt0} [deg.]')
    update(' ')
    update(f'--> <dt> .......... = {dt1} [deg.]')
    update(f'--> <predicted> ... = {dt0} [deg.]')

# Function to Indefinitely Bounce Motor (t) Between Extremal Limits
def bounce_t():
    """
    Moves system back and forth between extremal azimuthal 
    displacements indefinitely. 

    Parameters: 
    None

    Returns:
    None 

    """

    # Update console
    update('bouncing t ...')

    # Home t-motor
    home_t()

    # Prime t-motor
    advance_t(2000,vt_fast)

    # Bounce loop
    time.sleep(2)
    while True:
        advance_t(2000,vt_fast)
        time.sleep(1)
        advance_t(-2000,vt_fast)
        time.sleep(1)

# Function to Run Full Scanning Pass
def runPass(U,P,a=0,b=0):
    """
    Executes full scanning pass. 

    Parameters: 
    U (array): Global Parameters 
    P (array): Local ParametersLocal Parameters. 
    a (integer): Index of initial trace - for restarts [1]
    b (integer): Index of final trace - for restarts [1]

    Returns:
    None 

    """

    # Initialise console
    update(f'----- scan_id:{U[0]:0>3} -----')

    # Save copy of metadata
    dump_metadata(U,P)

    # x/t prioritisation
    scan_type = U[2]

    # Sound queues
    takeScan = False
    nextTransect = True

    # Compute number of scanning positions
    Nx = int(np.ceil((P[1]-P[0])/P[4]))+1
    Nt = int(np.ceil((P[3]-P[2])/P[5]))+1

    # convert dimensional parameters <P> to n_steps [1]
    nx0 = int(convert_lx2nx(P[0]))
    nx1 = int(convert_lx2nx(P[1]))
    nt0 = int(convert_lt2nt(P[2]))
    nt1 = int(convert_lt2nt(P[3]))
    ndx = int(convert_lx2nx(P[4]))
    ndt = int(convert_lt2nt(P[5]))

    # Home to datum
    update('homing...')
    home_x()
    home_t()

    # Movement cycle
    if (scan_type == 'tx'):

        # move to initial position
        update('priming...')
        advance_x(nx0 + b*(ndx),vx_vfast)     # x --> transects (b)
        advance_t(nt0 + a*(ndt),vt_vfast)     # t --> traces (a)

        # data collection
        i0 = b
        update('begin data collection...')
        for i in range(i0,Nx):                          
            t0 = datetime.now()
            if (i == i0):                               
                j0 = a
            else:
                j0 = 0
            for j in range(j0,Nt):                      # cycle t
                if (U[3]==True):
                    vnaScanNow(vnaIP,id,i,j)
                else:
                    playAlert()
                x = P[0] + i*P[4]
                t = P[2] + j*P[5]
                update(f'b = {i:04d}/{Nx-1:04d} ({x:06.3f} [m]) \
                       | a = {j:04d}/{Nt-1:04d} ({t:07.3f} [deg])')
                if (j<Nt-1):
                    advance_t(ndt,vt_fast)              # next t
            if (i<Nx-1):
                playAttention()
                advance_x(ndx,vx_fast)                  # next x
                home_t()
                advance_t(nt0,vt_fast)
            t1 = datetime.now()
            etc(t0,t1,i,Nx)
            playAttention()
    elif(scan_type == 'xt'):

        # move to initial position
        update('priming...')
        advance_x(nx0 + a*(ndx),vx_vfast)
        advance_t(nt0 + b*(ndt),vt_vfast)

        # data collection
        j0 = b
        update('begin data collection...')
        for j in range(j0,Nt):                          # cycle t 
            t0 = datetime.now()
            if (j == j0):                               
                i0 = a
            else:
                i0 = 0
            for i in range(i0,Nx):                      # cycle x
                if (U[3] == True):
                    vnaScanNow(vnaIP,id,i,j)
                else:
                    playAlert()
                x = P[0] + i*P[4]
                t = P[2] + j*P[5]
                update(f'b = {j:04d}/{Nt-1:04d} ({t:07.3f} [deg]) \
                       | a = {i:04d}/{Nx-1:04d} ({x:06.3f} [m])')
                if (i<Nx-1):
                    advance_x(ndx,vx_fast)              # next x
            if (j<Nt-1):
                playAttention()
                advance_t(ndt,vt_fast)                  # next t
            home_x()
            advance_x(nx0,vx_fast)
            t1 = datetime.now()
            etc(t0,t1,j,Nt)
            playAttention()
    else:
        playAlarm()
        raise ValueError(f'invalid pass prioritisation {U}')

# ---------------------------------------------------------
# SUBROUTINES | ANCILLARY 
# ---------------------------------------------------------

# Function to Update Log
def update(M):
    """
    Prints timestamped update text to console and logfile. 

    Parameters: 
    M (string): Message text

    Returns:
    None 

    """

    # Current Time
    t = datetime.now()

    # Update Console
    msg = f'{t} | {M}'
    print(msg)

    # Update Log File
    with open('log.txt','a') as file:
        file.write(msg + '\n')

# Function to Wait For User Input
def wait4user(M):
    """
    Waits until user presses any key before continuing code. 

    Parameters: 
    M (string): Message text

    Returns:
    None 

    """
    t = datetime.now()
    if (M == ''):
        M = 'press any key to continue...'
    input(f'{t} | {M}')

# Function to Wait For Specific Time Interval 
def wait4(T):
    """
    Waits for specific time interval before code continues. 

    Parameters: 
    T (float): Wait interval [s]

    Returns:
    None 

    """
    for i in range(T):
        M = f'--> {T-i} sec'
        update(M)
        time.sleep(1)

# Function to Check if a File Exists
def file_exists(fileName):
    """
    Checks if specified file exists in current working directory. 

    Parameters: 
    fileName (string): Name of file, including extension. 

    Returns:
    fileState (integer): Existence state. 
    filePath (string): Absolute path to file.  

    """
    current_directory = os.path.dirname(os.path.abspath(__file__))
    filePath = os.path.join(current_directory, fileName)
    fileState = os.path.exists(filePath)
    return fileState, filePath

# Function to Print Time Metadata For Each Transect  
def etc(t0,t1,n,N):
    """
    Prints elapsed runtime of last transect and estimated time for 
    pass completion.

    Parameters: 
    t0 (float): Initial timer count [s] 
    t1(float): Final timer count [s]
    n (integer): Current transect index [1]
    N (integer): Total number of transects [1] 

    Returns:
    None 

    """
    '''
    
    '''
    current_time = datetime.now()
    last_transect_runtime = t1-t0
    transects_left = ((N-1)-n)
    runtime_left = (transects_left)*(last_transect_runtime)
    finish_time = current_time + runtime_left
    M1 = f'b = {n}/{N-1} | ........................................\
        ............ transect duration = {last_transect_runtime}'
    M2 = f'b = {n}/{N-1} | ........................................\
        .......... pass completion due = {finish_time}'
    update(M1)
    update(M2)

# Function to Play Sound Files
def playSound(fileName):
    """
    Plays specified sound file.  

    Parameters: 
    fileName (string): Name of sound file including extension. 
    
    Returns:
    None

    """ 
    fileState, filePath = file_exists(fileName)
    if (fileState == False):
        print(f'Warning: (!) cannot find {fileName}')
    else:

        # play audio file
        pygame.init()
        pygame.mixer.init()
        pygame.mixer.music.load(filePath)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)

# Function to Play Alert Sound
def playAlert():
    """
    Plays alert sound.  

    Parameters: 
    None
    
    Returns:
    None

    """ 
    playSound('alert.mp3')

# Function to Play Alarm Sound
def playAlarm():
    """
    Plays alarm sound.  

    Parameters: 
    None
    
    Returns:
    None

    """ 
    playSound('alarm.mp3')

# Function to Play Attention Sound
def playAttention():
    """
    Plays attention sound.  

    Parameters: 
    None
    
    Returns:
    None

    """ 
    playSound('attention.mp3')

# Function to Play Done Sound
def playDone():
    """
    Plays done sound.  

    Parameters: 
    None

    Returns:
    None

    """ 
    playSound('done.mp3')

# Function to Read Limit Switch State 
def switchRead(S):
    """
    Reads switch state with debounce applied to suppress line noise. 
       
    Parameters: 
    S (string): Switch handle

    Returns:
    0: Switch pressed
    1: Switch not pressed

    """ 

    # configure debounce
    db_window = 50
    db_threshold = 38
    db_count = 0

    # read state (ply debounce)
    for i in range(db_window):
        switchState = GPIO.input(S)
        if (switchState ==  GPIO.LOW):
            db_count += 1
    if (db_count > db_threshold):
        return 0                            # switch pressed
    else:
        return 1                            # switch not pressed

# Function to Check For Danger Before Movement
def check4danger(H):
    """
    Checks if specified combinations of limit switches are triggered
    Halts system if danger detected.   

    Parameters: 
    H (string): Combination of switch states to check 
                     'any','min','max','xmin','xmax','tmin','tmax'

    Returns:
    None

    """        

    # Read limit switch states
    state_xMin = switchRead(limitSwitch1_xMin)
    state_xMax = switchRead(limitSwitch2_xMax)
    state_tMin = switchRead(limitSwitch3_tMin)
    state_tMax = switchRead(limitSwitch4_tMax)
    state_any = (state_xMin)*(state_xMax)*(state_tMin)*(state_tMax)
    state_min = (state_xMin)*(state_tMin)
    state_max = (state_xMax)*(state_tMax)

    # React (if required)
    handles = ['any','min','max','xmin','xmax','tmin','tmax']
    if ((H == handles[0]) and (state_any == 0)):
        playAlarm()
        raise RuntimeError('(!): unexpected <any> limit switch \
                           trip event --> all halt')
    elif ((H == handles[1]) and (state_min == 0)):
        playAlarm()
        raise RuntimeError('(!): unexpected <min> limit switch \
                           trip event --> all halt')
    elif ((H == handles[2]) and (state_max == 0)):
        playAlarm()
        raise RuntimeError('(!): unexpected <max> limit switch \
                           trip event --> all halt')
    elif ((H == handles[3]) and (state_xMin == 0)):
        playAlarm()
        raise RuntimeError('(!): unexpected <xMin> limit switch \
                           trip event --> all halt')
    elif ((H == handles[4]) and (state_xMax == 0)):
        playAlarm()
        raise RuntimeError('(!): unexpected <xMax> limit switch \
                           trip event --> all halt')
    elif ((H == handles[5]) and (state_tMin == 0)):
        playAlarm()
        raise RuntimeError('(!): unexpected <tMin> limit switch \
                           trip event --> all halt')
    elif ((H == handles[6]) and (state_tMax == 0)):
        playAlarm()
        raise RuntimeError('(!): unexpected <tMax> limit switch \
                           trip event --> all halt')
    elif H not in handles:
        playAlarm()
        raise ValueError('(!): invalid <check4danger> group handle')

# Function to Convert Lateral Displacement to Motor Step Count. 
def convert_lx2nx(lx):
    """
    Converts lateral displacement in [m] to number of motor steps.  

    Parameters: 
    lx (integer): Desired lateral displacement [m]

    Returns:
    nx (integer): Associated number of motor steps.  

    """    
    nx = lx/alpha
    return nx

# Function to Convert Azimuthal Displacement to Motor Step Count. 
def convert_lt2nt(lt):
    """
    Converts azimuthal displacement [deg] to number of motor steps.  

    Parameters: 
    lt (integer): Desired azimuthal displacement [deg]

    Returns:
    nt (integer): Associated number of motor steps.  

    """    
    nt = lt/beta
    return nt

# Function to Export Metadata to File.
def dump_metadata(U,P):
    """
    Copies python script to current working directory. 
    Copies parameters to file in current working directory. 

    Parameters: 
    U (array): Global Parameters 
    P (array): Local Parameters

    Returns:
    None 

    """
    
    # Get current working directory
    cwd = os.getcwd()

    # Copy python script
    fileName0 = 'SUNDIAL.py'
    fileName1 = f's{U[0]}_SUNDIAL.py'
    filePath0 = os.path.join(cwd,fileName0)
    filePath1 = os.path.join(cwd,fileName1)
    shutil.copy(filePath0, filePath1)
    del fileName0, fileName1, filePath0, filePath1

    # Copy parameters
    fileName = f's{U[0]}_parameters_scan.txt'
    filePath = os.path.join(cwd,fileName)
    np.savetxt(filePath, np.concatenate((U,P),axis=0), fmt='%s')

    # -------------------
    return []

# ---------------------------------------------------------
# MAIN 
# ---------------------------------------------------------
if __name__ == '__main__':

    # Metadata
    scan_id = '000'
    scan_tag = '3sphere_coverOn'
    scan_type = 'tx'

    # Dimensional Parameters
    x0 = 0.325                              # x-start .....[m]
    x1 = 0.985                              # x-end .......[m]
    t0 = 105-25.2                           # t-start .....[deg]
    t1 = 105+25.2                           # t-end .......[deg]
    dx = 0.01                               # x-inc .......[m]
    dt = 0.828                              # t-inc .......[deg]

    # Start Scan From...
    a = 0                                   # trace index
    b = 0                                   # transect index

    # Enable VNA Measurement
    enableVNA = True

    # Merge Parameters
    U = [scan_id,scan_tag,scan_type,enableVNA]
    P = [x0,x1,t0,t1,dx,dt]

    # List of Commands
    """
    runPass(U,P,a=0,b=0)
    home_x()
    home_t()
    max_x()
    max_t()
    advance_x(400,vx_fast)
    advance_t(10000,vt_fast)
    advance_t(-1000,vt_fast)
    bounce_x()
    bounce_t()
    playAlert()
    playAlarm()
    playAttention()
    playDone()
    """

    # Run Scanning Pass
    try:
        wait4user('press enter to begin...')
        time.sleep(5)
        runPass(U,P,a,b)
        playDone()
    except KeyboardInterrupt:
        playAlarm()
        GPIO.cleanup()
    finally:
        wait4user('press enter to finish...')
        update('complete')
        GPIO.cleanup()
