import vitaldb as vd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import filter_operation
import math_func as mf
import correlate_minmax as cmm
from calculate_stats import calculate_statistics
from data_processing import read_data, process_data, display_matrices, stacker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn import metrics
from correlate_minmax import corrMaxMain, corrMaxMain2
from data_prediction import predict, plotRes, predict3, predict4


def datazo(dir):
    testObj = vd.read_vital("VitalDB_data/VitalDB_data/1.vital")
    test = testObj.get_track_names()
    print(test)
    setD = read_data(test, testObj)
    M_SPO2, M_ART, M_PLETH = process_data(setD,'Infinity/PLETH_SPO2', 'SNUADC/ART', 'SNUADC/PLETH')
    print(len(M_PLETH))
    print(len(M_ART))
    return M_ART, M_PLETH

def filtering(M_ART,M_PLETH):

    sigma = 5
    k=0
    # ART
    filtered_M_ART, k, locAbsM_ART, locAbsm_ART, smoothedM_Art = filter_operation.apply_gaussian_filter(M_ART,"M_ART", sigma, k)
    # PLETH
    filtered_M_PLETH, k, locAbsM_PLETH, locAbsm_PLETH, smoothedM_Pleth = filter_operation.apply_gaussian_filter(M_PLETH, "M_PLETH", sigma, k)

    maximumsVectPLETHT, minimumsVectPLETHT, maximumsPosPLETH, minimumsPosPLETH = cmm.corrMax2(filtered_M_PLETH[1000000:1006000])
    maximumsVectARTT, minimumsVectARTT, maximumsPosART, minimumsPosART = cmm.corrMax2(filtered_M_ART[1000000:1006000])


    plt.figure()
    plt.plot(M_ART[1000000:1006000])
    plt.title("Received ART")
    plt.show(block = False)

    plt.figure()
    plt.plot(M_PLETH[1000000:1006000])
    plt.title("Received PLETH")
    plt.show(block = False)

    plt.figure()
    plt.plot(M_ART[1000000:1006000], '-b', alpha=0.5, label = 'PRE-Filter ART')
    plt.plot(filtered_M_ART[1000000:1006000], '-r', label= 'Filtered ART')
    plt.title("Filtered vs Non-Finltered ART")
    plt.legend()
    plt.show(block = False)

    plt.figure()
    plt.plot(M_ART[1000000:1006000], '-b', alpha=0.5, label = 'PRE-Filter ART')
    plt.plot(filtered_M_ART[1000000:1006000], '-r', label= 'Filtered ART')
    plt.scatter(maximumsPosART,maximumsVectARTT, marker='X', c='orange')
    plt.scatter(minimumsPosART,minimumsVectARTT, marker='X', c='orange')
    plt.title("Filtered vs Non-Finltered ART")
    plt.legend()
    plt.show(block = False)

    plt.figure()
    plt.plot(M_PLETH[1000000:1006000], '-b', alpha=0.5, label = 'PRE-Filter PLETH')
    plt.plot(filtered_M_PLETH[1000000:1006000], '-r', label= 'Filtered PLETH')
    plt.legend()
    plt.title("Filtered vs Non-Finltered PLETH")
    plt.show(block = False)

    plt.figure()
    plt.plot(M_PLETH[1000000:1006000], '-b', alpha=0.5, label = 'PRE-Filter PLETH')
    plt.plot(filtered_M_PLETH[1000000:1006000], '-r', label= 'Filtered PLETH')
    plt.scatter(maximumsPosPLETH,maximumsVectPLETHT, marker='X', c='orange')
    plt.scatter(minimumsPosPLETH,minimumsVectPLETHT, marker='X', c='orange')
    plt.legend()
    plt.title("Filtered vs Non-Finltered PLETH")
    plt.show(block = False)

    maxvectPLETH, maxvectART, minvectPLETH, minvectART, maximumsPosPLETH, maximumsPosART, minimumsPosPLETH, minimumsPosART, frecPLETH, frecART, intPLETH, intART  = corrMaxMain2(filtered_M_PLETH[0:],filtered_M_ART[0:],2,20,40000)

    n1 = len(maxvectPLETH)
    n2 = len(minvectPLETH)
    n3 = min(n1,n2)
    maxvectPLETH_PRED = np.zeros((n3,4))
    maxvectPLETH_PRED[:,0] = maxvectPLETH[:n3]
    maxvectPLETH_PRED[:,1] = minvectPLETH[:n3]
    maxvectPLETH_PRED[:,2] = frecPLETH[:n3]
    maxvectPLETH_PRED[:,3] = intPLETH[:n3]

    plt.figure()
    plt.plot(maxvectPLETH_PRED[:,0], '-b')
    plt.title("PLETH cycle maximums")
    plt.show(block = False)

    plt.figure()
    plt.plot(maxvectPLETH_PRED[:,1], '-b')
    plt.title("PLETH cycle minimums")
    plt.show(block = False)

    plt.figure()
    plt.plot(maxvectPLETH_PRED[:,2], '-b')
    plt.title("PLETH cycle duration")
    plt.show(block = False)

    plt.figure()
    plt.plot(maxvectPLETH_PRED[:,3], '-b')
    plt.title("PLETH cycle integral")
    plt.show(block = False)

    plt.figure()
    plt.plot(maxvectART, '-b')
    plt.title("ART cycle maximums")
    plt.show(block = False)

    plt.figure()
    plt.plot(minvectART, '-b')
    plt.title("ART cycle minimums")
    plt.show(block = False)


    print("::::::::")
    print(maxvectPLETH_PRED)

    plt.figure()
    plt.plot(maxvectPLETH_PRED[:,3])
    plt.title("PLETH CYCLE INTEGRALS")
    plt.show(block=False)
    predictions_max, maxtest = predict(maxvectPLETH_PRED, maxvectART[:n3],False,False) # predict x from y 

    print("------------MINIMUMS--------------")

    # Predictions min :
    minvectPLETH_PRED = np.zeros((n3,4))
    minvectPLETH_PRED[:,0] = minvectPLETH[:n3]
    minvectPLETH_PRED[:,1] = maxvectPLETH[:n3]
    maxvectPLETH_PRED[:,2] = frecPLETH[:n3]
    maxvectPLETH_PRED[:,3] = intPLETH[:n3]
    predictions_min, mintest = predict(minvectPLETH_PRED, minvectART[:n3],False,False) # predict x from y 

    # plotRes(predictions_max,maxtest)
    # plotRes(predictions_min,mintest)

    plt.axis([0, 20, 30, 120])
    y = []
    z = []
    plt.figure()
    plt.title("ART Min predictions")
    plt.xlabel("Cycle")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    print(" ")
    for i in range(50):
        y = np.append(y,predictions_min["GradientBoosting"][i])
        z = np.append(z,mintest[i])
        plt.plot(np.arange(i+1), y, '-b', label="Prediction")
        plt.plot(np.arange(i+1), z, '-r', label="Actual value")
        plt.pause(0.5)
        print("time:", time.ctime(time.time()),"max pred:",predictions_max["GradientBoosting"][i],"min pred:",predictions_min["GradientBoosting"][i],sep=" ")
    plt.show()


M_ART, M_PLETH = datazo(None)
filtering(M_ART=M_ART, M_PLETH=M_PLETH)
plt.show()

