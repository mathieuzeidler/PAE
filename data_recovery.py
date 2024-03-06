import vitaldb as vd
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


###############################################################################################################

    ## Local smoothing of data

###############################################################################################################
    

#JUST AN IDEA, WORK IN PROGRESS
def windowedSmoothing(x,winSize,tol,minSD):
    i = len(x)
    length = len(x)
    smoothedSig = np.array([])
    while i-winSize>=0:
        meanML = np.sum(x[length-i:length-i+winSize])/winSize
        sdML = np.sqrt(np.sum((x[length-i:length-i+winSize]-meanML)**2)/(winSize-1))
        sdML = sdML + minSD*(sdML<tol)
        smoothedSig = np.concatenate((smoothedSig,gaussian_filter1d(x[length-i:length-i+winSize],sigma=sdML,mode='reflect')))
        i = i-winSize
    if i>0:
        smoothedSig = np.concatenate((smoothedSig,windowedSmoothing(x[length-i:],len(x[length-i:]),tol,minSD)))
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



# def localMaxPos(dx, tol):
#     criticalIndex = np.array([])
#     maxIndex = np.array([])
#     maxValues = np.array([])
#     minIndex = np.array([])
#     minValues = np.array([])
#     for i in range(len(dx)):
#         if abs(dx[i])<tol:
#             criticalIndex = np.concatenate((criticalIndex,i))
#     for j in range(len(criticalIndex)):
        
        
    

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
    plt.plot(filtered_M_ART[1000000:1050001], label='Filtered ART')
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
    plt.plot(windowedSmoothing(M_ART[1000000:1050001],1000,0.1,4),label="WINDOWED FILTER ART")
    plt.xlabel("t")
    plt.ylabel("SNUADC/ART (Filtered)")
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



