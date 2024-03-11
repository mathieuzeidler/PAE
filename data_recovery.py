import vitaldb as vd
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import symiirorder2
from scipy.signal import argrelextrema


###############################################################################################################

    ## Local smoothing of data

###############################################################################################################
    

#JUST AN IDEA, WORK IN PROGRESS
def windowedSmoothing(x, winSize, tol, minSD):
    i = len(x)
    length = len(x)
    smoothedSig = np.array([])
    semiWin = round(winSize/2)
    while i - winSize >= 0:
        imin = length - i;
        imax = length - i + winSize;
        current_slice = x[imin:imax]
        #valid_values = current_slice[~np.isnan(current_slice)] we shoukd filter out NaN values before filtering
        valid_values = current_slice
        if imin>0:
            valid_values = np.concatenate((x[imin-semiWin:imin],valid_values))
        upperWin = min(semiWin,length-imax)
        if upperWin>0:
            valid_values = np.concatenate((valid_values,x[imax:imax+upperWin]))
        if len(valid_values) > 0:
            sdML = np.std(valid_values)
            
            if sdML > 0:
                sdML = sdML + minSD * (sdML < tol)
                smoothed_slice = gaussian_filter1d(valid_values, sigma=sdML, mode='reflect')
                if imin>0:
                    smoothedSig = np.concatenate((smoothedSig, smoothed_slice[semiWin:semiWin+imax-imin]))
                else:
                    smoothedSig = np.concatenate((smoothedSig, smoothed_slice[:imax]))
        i = i - winSize

    if i > 0:
        smoothedSig = np.concatenate((smoothedSig, windowedSmoothing(x[length - i:], len(x[length - i:]), tol, minSD)))

    return smoothedSig

###############################################################################################################

    ## Derivative

###############################################################################################################

def finiteDiffDiscrete(x):
    dx = np.zeros((len(x)))
    if len(x)>2:
        for i in range(len(x)-1):
            dx[i]=(x[i+1]-x[i])
            dx[-1] = dx[-3]
    return dx

def finiteDiffIndexed(x,t):
    dx = np.zeros((len(x)))
    if len(x)>2:
        for i in range(len(x)-1):
            dx[i] = (x[i+1]-x[i])/(t[i+1]-t[i])
        dx[-1] = dx[-2]

    return dx

###############################################################################################################

    ## Max/Min detection

###############################################################################################################


#GO-TO function
def localMaxOP2(x):
    return argrelextrema(x,np.greater)

#If the first method did not find a specific loc,Max, use this function
def localMaxPos(dx, tol):
    criticalIndex = np.where(abs(dx)<tol)
    return criticalIndex

def localMinOP2(x):
    return argrelextrema(x,np.less)

    

###############################################################################################################

    ## Reading the data localy and create the matrices to store the values

###############################################################################################################

## For now we just read one file localy but we will need to read more files and in a continue way
DOWNLOAD_DIR = "VitalDB_data/VitalDB_data/NewData"
if not os.path.exists(DOWNLOAD_DIR):
    print("Creating new directory")
    os.mkdir(DOWNLOAD_DIR)

# these are data from Ramses
testObj = vd.read_vital("VitalDB_data/VitalDB_data/1.vital")

# these are data from VitalDB OpenData Set
#testObj = vd.read_vital("VitalDB_data/VitalDB_data/1.vital") 


test = testObj.get_track_names()
setD = list()

# Create matrices to store the values (can add more variables)
M_SPO2 = np.array([])
M_ART = np.array([])
M_PLETH = np.array([])

for f in test:
    setD.append((f, testObj.to_numpy(f, None)))

#################################################################################################################
    
    ## Displaying the data and storing the values in the matrices

#################################################################################################################

k = 1
pleth_spo2_found = False
art_found = False
pleth_wf_found = False

## Take only 2 hours on vitaldb opendata set
for dataPair in setD:
    if dataPair[0] == 'Infinity/PLETH_SPO2' or dataPair[0] == 'SNUADC/ART' or dataPair[0] == "SNUADC/PLETH":
        # x = np.arange(0, len(dataPair[1]))
        # #print(dataPair[1]) # Uncomment to see the values of the data
        # plt.figure(k)
        # plt.stem(x, dataPair[1], linefmt='grey', markerfmt='')
        # plt.xlabel("t")
        # plt.ylabel(dataPair[0])
        # plt.title(dataPair[0])
        # k = k + 1

        if dataPair[0] == 'Infinity/PLETH_SPO2':
            pleth_spo2_found = True
            # Adding values to the M_SPO2 matrix without NaN
            values = dataPair[1][~np.isnan(dataPair[1])]
            if len(values) > 0:
                if M_SPO2.size == 0:
                    M_SPO2 = values
                else:
                    M_SPO2 = np.vstack((M_SPO2, values))
                    
        elif dataPair[0] == 'SNUADC/ART':
            art_found = True
            # Adding values to the M_ART matrix without NaN
            values = dataPair[1][~np.isnan(dataPair[1])]
            if len(values) > 0:
                if M_ART.size == 0:
                    M_ART = values
                else:
                    M_ART = np.vstack((M_ART, values))
        elif dataPair[0] == "SNUADC/PLETH":
            pleth_wf_found = True
            values = dataPair[1][~np.isnan(dataPair[1])]
            if len(values) > 0:
                if M_PLETH.size == 0:
                    M_PLETH = values
                else:
                    M_PLETH = np.vstack((M_SPO2, values))

if not pleth_spo2_found:
    print("There is not PLETH_SPO2 in the data")

if not art_found:
    print("There is not ART in the data")

# uncomment to show the plots
#plt.show()

# Displaying the M matrices
if M_SPO2.size > 0:
    print("Matrix M_SPO2:")
    print(M_SPO2)

if M_ART.size > 0:
    print("\nMatrix M_ART:")
    print(M_ART)

###############################################################################################################
    
    ## Apply Gaussian filter to ART signal

###############################################################################################################
    
# Filter parameters
sigma = 5  # We can adjust it

# Apply Gaussian filter to ART signal
if M_ART.size > 0:
    filtered_M_ART = gaussian_filter1d(M_ART, sigma=sigma, mode='reflect')

    # Display the filtered ART signal
    plt.figure(k)
    plt.plot(filtered_M_ART[1000000:1002001], label='Filtered ART')
    plt.xlabel("t")
    plt.ylabel("SNUADC/ART (Filtered)")
    plt.title("Filtered SNUADC/ART Signal")
    plt.legend()
    k += 1

    # Display the original and filtered signals
    plt.figure(k)
    plt.plot(M_ART, label='Original ART', alpha=0.5)
    plt.plot(filtered_M_ART, label='Filtered ART', linewidth=2)
    plt.xlabel("t")
    plt.ylabel("SNUADC/ART")
    plt.title("Original vs. Filtered SNUADC/ART Signal")
    plt.legend()
    k += 1
    
    plt.figure(k)
    smoothedM_Art = windowedSmoothing(M_ART[2002000:2004001],200,0.1,4)
    dxM_Art = finiteDiffDiscrete(smoothedM_Art)
    critM_Art = localMaxPos(dxM_Art,0.5e-2)
    locMaxM_Art = localMaxOP2(smoothedM_Art)
    locMinM_Art = localMinOP2(smoothedM_Art)
    plt.plot(smoothedM_Art,label="WINDOWED FILTER ART")
    #plt.vlines(critM_Art,30,98,colors='lightcoral')
    plt.vlines(locMaxM_Art,30,98,colors='green')
    plt.vlines(locMinM_Art,30,98,colors='yellow')
    plt.xlabel("t")
    plt.ylabel("SNUADC/ART (Filtered)")
    plt.legend()
    k += 1
    
    plt.figure(k)
    plt.plot(smoothedM_Art[720:820],label="WINDOWED FILTER ART")
    plt.xlabel("t")
    plt.ylabel("SNUADC/ART (Filtered)")
    plt.legend()
    k += 1
    
    plt.figure(k)
    plt.plot(smoothedM_Art,label="WINDOWED FILTER ART")
    plt.xlabel("t")
    plt.ylabel("SNUADC/ART (Filtered)")
    plt.legend()
    k += 1
    
    plt.figure(k)
    plt.plot(dxM_Art,label="DX WINDOWED FILTER ART")
    plt.plot(finiteDiffDiscrete(dxM_Art),label="DXDX WINDOWED FILTER ART")
    plt.plot()
    plt.plot
    plt.xlabel("t")
    plt.ylabel("SNUADC/ART (Derivative)")
    plt.legend()
    k += 1
    
    plt.figure(k)
    plt.plot(finiteDiffDiscrete(dxM_Art),label="DXDX WINDOWED FILTER ART")
    plt.plot()
    plt.plot
    plt.xlabel("t")
    plt.ylabel("SNUADC/ART (Derivative)")
    plt.legend()
    k += 1
    
    # Update M_ART with the filtered values
    M_ART = filtered_M_ART

###################################################################################################
    
    ## Calculating the minimum, maximum, and standard deviation for the matrices

#####################################################################################################

# Calculation of the minimum, maximum, and standard deviation for M_SPO2
if M_SPO2.size > 0:
    min_val_M_SPO2 = np.min(M_SPO2)
    max_val_M_SPO2 = np.max(M_SPO2)
    std_dev_M_SPO2 = np.std(M_SPO2)

    print(f"\nOverall Minimum of Matrix M_SPO2: {min_val_M_SPO2}")
    print(f"Overall Maximum of Matrix M_SPO2: {max_val_M_SPO2}")
    print(f"Overall Standard Deviation of Matrix M_SPO2: {std_dev_M_SPO2}")

# Calculation of the minimum, maximum, and standard deviation for M_ART
if M_ART.size > 0:
    min_val_M_ART = np.min(M_ART)
    max_val_M_ART = np.max(M_ART)
    std_dev_M_ART = np.std(M_ART)

    print(f"\nOverall Minimum of Matrix M_ART: {min_val_M_ART}")
    print(f"Overall Maximum of Matrix M_ART: {max_val_M_ART}")
    print(f"Overall Standard Deviation of Matrix M_ART: {std_dev_M_ART}")



