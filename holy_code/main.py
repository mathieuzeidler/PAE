import vitaldb as vd
import numpy as np
import os
import matplotlib.pyplot as plt
import filter_operation
import math_func as mf
from calculate_stats import calculate_statistics
from data_processing import read_data, process_data, display_matrices, stacker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn import metrics
from correlate_minmax import corrMaxMain, corrMaxMain2
from data_prediction import predict, plotRes, predict3, predict4

###############################################################################################################

    ## Reading the data localy and create the matrices to store the values

###############################################################################################################

imp = False

## For now we just read one file localy but we will need to read more files and in a continue way
if imp:
    DOWNLOAD_DIR = "VitalDB_data/VitalDB_data/NewData"
    if not os.path.exists(DOWNLOAD_DIR):
        print("Creating new directory")
        os.mkdir(DOWNLOAD_DIR)

    #testObj = vd.read_vital("VitalDB_data/VitalDB_data/1.vital")
    #testObj = vd.read_vital("VitalDB_data/19-3/QUI12_230718_175152.vital")
    testObj = vd.read_vital("VitalDB_data/9-4/240404/mj6uua9n3_240404_124253.vital")
    test = testObj.get_track_names()

    print(test)

    # Reading the data
    setD = read_data(test, testObj)

    #################################################################################################################
        
        ## Displaying the data and storing the values in the matrices

    #################################################################################################################

    k = 1
    M_SPO2, M_ART, M_PLETH = process_data(setD,'Infinity/PLETH_SPO2', 'Intellivue/ABP', 'Intellivue/PLETH')

    M_ART_STACKED, M_PLETH_STACKED = stacker(np.array(["VitalDB_data/9-4/240404/mj6uua9n3_240404_124253.vital","VitalDB_data/9-4/240404/mj6uua9n3_240404_101328.vital","VitalDB_data/9-4/240404/mj6uua9n3_240404_105021.vital","VitalDB_data/9-4/240404/mj6uua9n3_240404_164259.vital","VitalDB_data/9-4/240403/h48esuvf8_240403_165638.vital","VitalDB_data/9-4/240404/mj6uua9n3_240404_102619.vital","VitalDB_data/9-4/240406/mj6uua9n3_240406_073435.vital","VitalDB_data/9-4/240406/mj6uua9n3_240406_054613.vital","VitalDB_data/9-4/240405/mj6uua9n3_240405_212503.vital","VitalDB_data/9-4/240403/mj6uua9n3_240403_230953.vital"]))

    # Displaying the M matrices
    #display_matrices(M_SPO2, M_ART, M_PLETH) #uncomment to display the matrices

    ###############################################################################################################
        
        ## Apply Gaussian filter to ART signal

    ###############################################################################################################
        
    # Filter parameters
    sigma = 5  # We can adjust it

    print("MARSIZ:", M_ART.size)
    print("MAPLETH:", M_PLETH.size)

    # ART
    filtered_M_ART, k, locAbsM_ART, locAbsm_ART, smoothedM_Art = filter_operation.apply_gaussian_filter(M_ART_STACKED,"M_ART", sigma, k)
    # PLETH
    filtered_M_PLETH, k, locAbsM_PLETH, locAbsm_PLETH, smoothedM_Pleth = filter_operation.apply_gaussian_filter(M_PLETH_STACKED, "M_PLETH", sigma, k)
    # plt.show()
    # plt.figure()
    # plt.plot(filtered_M_ART)
    # plt.title("FILTERED ART")

    # plt.figure()
    # plt.plot(filtered_M_PLETH)
    # plt.title("FILTERED PLETH")
    # plt.show()


    #Correlations:
    #700000
    maxvectPLETH, maxvectART, minvectPLETH, minvectART, maximumsPosPLETH, maximumsPosART, minimumsPosPLETH, minimumsPosART, frecPLETH, frecART, intPLETH, intART  = corrMaxMain2(filtered_M_PLETH[0:],filtered_M_ART[0:],1.5,40,40000)


    # frequencyPLETH = np.copy(minimumsPosPLETH)

    # for i in range(len(frequencyPLETH)-1):
    #     frequencyPLETH[i] = frequencyPLETH[i+1]-frequencyPLETH[i]
    # print("**********+")
    # print(frequencyPLETH)
    # print("***********")

    # frequencyPLETH = frequencyPLETH[:-1]

    # pilot = np.load("PILOT.npy")

    # pilotInter = CubicHermiteSpline(np.arange(len(pilot)),pilot,mf.finiteDiffDiscrete(pilot))

    # plt.figure()
    # plt.plot(pilot)
    # xx = np.linspace(0,len(pilot),1000)
    # plt.plot(xx,pilotInter(xx))
    # plt.title("PILOT PLETH")
    # plt.show(block=False)


    # plt.figure()
    # plt.plot(np.correlate(pilot,filtered_M_PLETH))
    # plt.title("PILOT X-CORR")
    # plt.show(block = False)

    print(filtered_M_ART)

    plt.figure()
    plt.plot(maximumsPosPLETH)
    plt.title("PLETH Max Pos")
    plt.show(block=False)

    plt.figure()
    plt.plot(minimumsPosPLETH)
    plt.title("PLETH MIN Pos")
    plt.show(block=False)

    plt.figure()
    plt.plot(maxvectPLETH)
    plt.plot(maxvectART)
    plt.title("MAXIMUMS")
    plt.show(block=False)

    plt.figure()
    plt.plot(minvectPLETH)
    plt.plot(minvectART)
    plt.title("MAXIMUMS")
    plt.show(block=False)


    plt.figure()
    plt.plot(frecPLETH)
    plt.title("PLETH frec")
    plt.show(block=False)

    plt.show()

    # print("++++++++++++")
    # print(maximumsPosART)
    # print("++++++++++++")

    plt.figure()
    plt.plot(filtered_M_ART)
    plt.title("ART")
    plt.show(block=False)

    plt.figure()
    plt.plot(filtered_M_PLETH)
    plt.title("PLETH")
    plt.show(block=False)


    plt.figure()
    plt.plot(maxvectART)
    plt.title("maxvertART")
    plt.show(block=False)

    plt.figure()
    plt.plot(maxvectPLETH)
    plt.title("maxvertPLETH")
    plt.show(block=False)

    plt.figure()
    plt.plot(minvectART)
    plt.title("minvertART")
    plt.show(block=False)

    plt.figure()
    plt.plot(minvectPLETH)
    plt.title("minvertPLETH")
    plt.show(block=False)

    ###################################################################################################
    # Malak parts

    # Gradient Boosting Regression model to predict the correlation between the ART and PLETH signals
    #maxvectY_array = np.array(maxvectY).reshape(-1, 1)
    #model = GradientBoostingRegressor(n_estimators=100, max_depth=8)
    #model.fit(maxvectY_array, maxvectX)
    # R-squared score: 0.28817222879730764 for n_estimators=100, max_depth=3
    # R-squared score: 0.4778262112518894 for n_estimators=100, max_depth=5
    # R-squared score: 0.7249963779958533 for n_estimators=100, max_depth=8
    # R-squared score: 0.9248403061986639 for n_estimators=100, max_depth=13

    # Predict the values of cutMaximumsVectY based on cutMaximumsVectX_poly
    #prediction = model.predict(maxvectY_array)
    # Calculate the R-squared score
    #r2 = r2_score(maxvectX, prediction)
    #print("R-squared score:", r2)

    #maxvectY_array = np.array(maxvectY).reshape(-1, 1)
    #model = GradientBoostingRegressor(n_estimators=100,max_depth=13)
    #model.fit(maxvectY_array,maxvectX)
    #prediction = model.predict(maxvectY_array)
    #r2 = metrics.r2_score(maxvectX,prediction)
    #print("R-squared score:",r2)

    ###################################################################################################


    # Predictions max :
    n1 = len(maxvectPLETH)
    n2 = len(minvectPLETH)
    n3 = min(n1,n2)
    maxvectPLETH_PRED = np.zeros((n3,4))
    maxvectPLETH_PRED[:,0] = maxvectPLETH[:n3]
    maxvectPLETH_PRED[:,1] = minvectPLETH[:n3]
    maxvectPLETH_PRED[:,2] = frecPLETH[:n3]
    maxvectPLETH_PRED[:,3] = intPLETH[:n3]
    np.save("maxvertpleth",maxvectPLETH_PRED)
    np.save("maxvertART",maxvectART[:n3])

    plt.figure()
    plt.plot(maxvectPLETH_PRED[:,3])
    plt.title("PLETH CYCLE INTEGRALS")
    plt.show(block=False)
    predictions_max, maxtest = predict3(maxvectPLETH_PRED, maxvectART[:n3]) # predict x from y 

    print("------------MINIMUMS--------------")

    # Predictions min :
    minvectPLETH_PRED = np.zeros((n3,4))
    minvectPLETH_PRED[:,0] = minvectPLETH[:n3]
    minvectPLETH_PRED[:,1] = maxvectPLETH[:n3]
    maxvectPLETH_PRED[:,2] = frecPLETH[:n3]
    maxvectPLETH_PRED[:,3] = intPLETH[:n3]
    predictions_min, mintest = predict3(minvectPLETH_PRED, minvectART[:n3]) # predict x from y 

    np.save("minvertpleth",minvectPLETH_PRED)
    np.save("minvertart",minvectART[:n3])

    plotRes(predictions_max,maxtest)
    plotRes(predictions_min,mintest)

    plt.show()

    # Print predictions
    #print(predictions)

    # Plot predictions max
    plt.figure(figsize=(10, 6))
    plt.plot(predictions_max['LinearRegression'], 'r-')  # 'r-' means red line
    plt.plot(maxtest,'b-')
    plt.title('Predictions max of ART from PLETH')
    plt.xlabel('Time')
    plt.ylabel('Predicted Value')
    plt.show(block=False)


    # Plot predictions min
    plt.figure(figsize=(10, 6))
    plt.plot(predictions_min['LinearRegression'], 'r-')  # 'r-' means red line
    plt.plot(mintest,'b-')
    plt.title('Predictions min of ART from PLETH')
    plt.xlabel('Time')
    plt.ylabel('Predicted Value')
    plt.show(block=False)
else:
    maxvectPLETH_PRED = np.load("maxvertpleth.npy")
    maxvectART = np.load("maxvertART.npy")
    minvectPLETH_PRED = np.load("minvertpleth.npy")
    minvectART = np.load("minvertart.npy")

    predictions_max, maxtest = predict4(maxvectPLETH_PRED, maxvectART) # predict x from y 
    predictions_min, mintest = predict4(minvectPLETH_PRED, minvectART) # predict x from y 

    plotRes(predictions_max,maxtest)
    plotRes(predictions_min,mintest)
    plt.show()

    


###################################################################################################

# Get the y-values (ordinates) at the local maxima and minima for ART
locAbsM_values_ART = [[smoothedM_Art[int(t)] for t in row] for row in locAbsM_ART]
locAbsm_values_ART = [[smoothedM_Art[int(t)] for t in row] for row in locAbsm_ART]

# Get the y-values (ordinates) at the local maxima and minima for PLETH
locAbsM_values_PLETH = [[smoothedM_Pleth[int(t)] for t in row] for row in locAbsM_PLETH]
locAbsm_values_PLETH = [[smoothedM_Pleth[int(t)] for t in row] for row in locAbsm_PLETH]

#ART
#print("\nlocal minimum t ART: " + str(locAbsm_ART))
#print("local minimum value ART: " + str(locAbsm_values_ART))
#print("\nlocal maximum t ART: " + str(locAbsM_ART))
#print("local maximum value ART: " + str(locAbsM_values_ART))

#PLETH
#print("\nlocal minimum t PLETH: " + str(locAbsm_PLETH))
#print("local minimum value PLETH: " + str(locAbsm_values_PLETH))
#print("\nlocal maximum t PLETH: " + str(locAbsM_PLETH))
#print("local maximum value PLETH: " + str(locAbsM_values_PLETH))

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
#plt.close()
