import vitaldb as vd
import numpy as np
import os
import matplotlib.pyplot as plt
import filter_operation
from calculate_stats import calculate_statistics
from data_processing import read_data, process_data, display_matrices

###############################################################################################################

    ## Reading the data localy and create the matrices to store the values

###############################################################################################################

## For now we just read one file localy but we will need to read more files and in a continue way
DOWNLOAD_DIR = "VitalDB_data/VitalDB_data/NewData"
if not os.path.exists(DOWNLOAD_DIR):
    print("Creating new directory")
    os.mkdir(DOWNLOAD_DIR)

testObj = vd.read_vital("VitalDB_data/VitalDB_data/1.vital")
test = testObj.get_track_names()

# Reading the data
setD = read_data(test, testObj)

#################################################################################################################
    
    ## Displaying the data and storing the values in the matrices

#################################################################################################################

k = 1
M_SPO2, M_ART, M_PLETH = process_data(setD)


# Displaying the M matrices
display_matrices(M_SPO2, M_ART)

###############################################################################################################
    
    ## Apply Gaussian filter to ART signal

###############################################################################################################
    
# Filter parameters
sigma = 5  # We can adjust it

filtered_M_ART, k = filter_operation.apply_gaussian_filter(M_ART, sigma, k)


###################################################################################################
    
    ## Calculating the minimum, maximum, and standard deviation for the matrices

#####################################################################################################

calculate_statistics(M_SPO2, "M_SPO2")
calculate_statistics(M_ART, "M_ART")

###################################################################################################
    
    ## Calculate the integral of the filtred signal (trapz method) between max1 and max2

#####################################################################################################
    
max1 = 365
max2 = 805
if M_ART.size > 0 and max1 < max2:
    sliced_M_ART = M_ART[max1:max2]
    integral_M_ART = np.trapz(sliced_M_ART)
    print(f"\nIntegral of the filtered ART signal between max1 and max2: {-integral_M_ART}")

# comment to not display the plots
plt.show()
