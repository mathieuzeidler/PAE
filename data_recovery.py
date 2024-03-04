import vitaldb as vd
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

###############################################################################################################

    ## Reading the data localy and create the matrices to store the values

###############################################################################################################

## For now we just read one file localy but we will need to read more files and in a continue way
DOWNLOAD_DIR = "VitalDB_data/VitalDB_data/NewData"
if not os.path.exists(DOWNLOAD_DIR):
    print("Creating new directory")
    os.mkdir(DOWNLOAD_DIR)

testObj = vd.read_vital("VitalDB_data/VitalDB_data/230602/QUI12_230602_194231.vital")
test = testObj.get_track_names()
setD = list()

# Create matrices to store the values (can add more variables)
M_SPO2 = np.array([])
M_ART = np.array([])

for f in test:
    setD.append((f, testObj.to_numpy(f, None)))

#################################################################################################################
    
    ## Displaying the data and storing the values in the matrices

#################################################################################################################

k = 1
pleth_spo2_found = False
art_found = False

## Take only 2 hours on vitaldb opendata set
for dataPair in setD:
    if dataPair[0] == 'Infinity/PLETH_SPO2' or dataPair[0] == 'SNUADC/ART':
        x = np.arange(0, len(dataPair[1]))
        #print(dataPair[1]) # Uncomment to see the values of the data
        plt.figure(k)
        plt.stem(x, dataPair[1], linefmt='grey', markerfmt='')
        plt.xlabel("t")
        plt.ylabel(dataPair[0])
        plt.title(dataPair[0])
        k = k + 1

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

print("I AM ABLE")