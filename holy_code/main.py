import vitaldb as vd
import numpy as np
import os
import matplotlib.pyplot as plt
import filter_operation
from calculate_stats import calculate_statistics
from data_processing import read_data, process_data, display_matrices
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from correlate_minmax import corrMaxMain
from data_prediction import predict

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
#display_matrices(M_SPO2, M_ART) uncomment to display the matrices

###############################################################################################################
    
    ## Apply Gaussian filter to ART signal

###############################################################################################################
    
# Filter parameters
sigma = 5  # We can adjust it

# ART
filtered_M_ART, k, locAbsM_ART, locAbsm_ART, smoothedM_Art = filter_operation.apply_gaussian_filter(M_ART,"M_ART", sigma, k)
# PLETH
filtered_M_PLETH, k, locAbsM_PLETH, locAbsm_PLETH, smoothedM_Pleth = filter_operation.apply_gaussian_filter(M_PLETH, "M_PLETH", sigma, k)

#Correlations:
maxvectY, maxvectX = corrMaxMain(M_ART,M_PLETH,2000,sigma) # y => ART, x => PLETH

###################################################################################################

#Predictions:
maxvectX = np.array(maxvectX).reshape(-1, 1) # reshape the data
predictions = predict(maxvectX, maxvectY) # predict x from y 

# Print predictions
#print(predictions)

# Plot predictions
plt.figure(figsize=(10, 6))
plt.plot(predictions, 'r-')  # 'r-' means red line
plt.title('Predictions of ART from PLETH')
plt.xlabel('Time')
plt.ylabel('Predicted Value')
plt.show()

###################################################################################################

# Get the y-values (ordinates) at the local maxima and minima for ART
locAbsM_values_ART = [[smoothedM_Art[int(t)] for t in row] for row in locAbsM_ART]
locAbsm_values_ART = [[smoothedM_Art[int(t)] for t in row] for row in locAbsm_ART]

# Get the y-values (ordinates) at the local maxima and minima for PLETH
locAbsM_values_PLETH = [[smoothedM_Pleth[int(t)] for t in row] for row in locAbsM_PLETH]
locAbsm_values_PLETH = [[smoothedM_Pleth[int(t)] for t in row] for row in locAbsm_PLETH]

#ART
print("\nlocal minimum t ART: " + str(locAbsm_ART))
print("local minimum value ART: " + str(locAbsm_values_ART))
print("\nlocal maximum t ART: " + str(locAbsM_ART))
print("local maximum value ART: " + str(locAbsM_values_ART))

#PLETH
print("\nlocal minimum t PLETH: " + str(locAbsm_PLETH))
print("local minimum value PLETH: " + str(locAbsm_values_PLETH))
print("\nlocal maximum t PLETH: " + str(locAbsM_PLETH))
print("local maximum value PLETH: " + str(locAbsM_values_PLETH))

###################################################################################################
    
    ## Calculating the minimum, maximum, and standard deviation for the matrices

#####################################################################################################

calculate_statistics(smoothedM_Pleth, "M_PLETH")

calculate_statistics(smoothedM_Art, "M_ART")

###################################################################################################

    ## Check if the patient is in good health

###################################################################################################

# Check if max_val is greater than 140 or min_val is less than 90
# Flatten the lists
flat_locAbsM_values = [item for sublist in locAbsM_values_ART for item in sublist]
flat_locAbsm_values = [item for sublist in locAbsm_values_ART for item in sublist]

# Check if any value in flat_locAbsM_values is greater than 140
dangerous_max_values = [value for value in flat_locAbsM_values if value > 140]
if dangerous_max_values:
    print('---------------------------------------------------------------------------------- ')
    print(' ')
    print(f"Danger ! The following maximum values are greater than 140: {dangerous_max_values}")
    print(' ')
    print('---------------------------------------------------------------------------------- ')

# Check if any value in flat_locAbsm_values is less than 50
dangerous_min_values = [value for value in flat_locAbsm_values if value < 50]
if dangerous_min_values:
    print('---------------------------------------------------------------------------------- ')
    print(' ')
    print(f"Danger ! The following minimum values are less than 50: {dangerous_min_values}")
    print(' ')
    print('---------------------------------------------------------------------------------- ')

###################################################################################################
    
    ## Calculate the integral of the filtred signal (trapz method) between max1 and max2 for ART

#####################################################################################################
    
max1 = 365
max2 = 805
if M_ART.size > 0 and max1 < max2:
    sliced_M_ART = smoothedM_Art[max1:max2]
    integral_M_ART = np.trapz(sliced_M_ART)
    print(f"\nIntegral of the filtered ART signal between max1 and max2: {integral_M_ART}")

###################################################################################################
    
    ## Calculate the integral of the filtred signal (trapz method) between max1 and max2 for PLETH

#####################################################################################################
    
max1 = 180
max2 = 610
if M_ART.size > 0 and max1 < max2:
    sliced_M_PLETH = smoothedM_Pleth[max1:max2]
    integral_M_PLETH = np.trapz(sliced_M_PLETH)
    print(f"\nIntegral of the filtered PLETH signal between max1 and max2: {integral_M_PLETH}")

# Ajuster automatiquement les paramètres de mise en page pour éviter le chevauchement
plt.tight_layout()

# Afficher le plot
plt.show()
plt.close()
