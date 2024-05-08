import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
from scipy.integrate import simps
from scipy.interpolate import CubicHermiteSpline


###############################################################################################################

    ## Local smoothing of data

###############################################################################################################
    
#def generatePILOT(T,max,min,integral):

def integrate(x, posX):
    integral = [0]
    for i in range(len(posX)-1):
        if len(x[posX[i]:posX[i+1]+1])==0:
            integral.append(0)
        else:
            integral.append(simps(x[posX[i]:posX[i+1]+1]))
    integral = np.array(integral)
    return integral

#JUST AN IDEA, WORK IN PROGRESS
def windowedSmoothingProcess(x, winSize, tol, minSD):
    i = len(x)
    length = len(x)
    smoothedSig = np.array([])
    # semiWin = round(winSize/2)
    semiWin = winSize
    while i - winSize >= 0:
        imin = length - i
        imax = length - i + winSize
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
                    smoothedSig = np.concatenate((smoothedSig, smoothed_slice[semiWin+1:semiWin+1+imax-imin]))
                else:
                    smoothedSig = np.concatenate((smoothedSig, smoothed_slice[:imax]))
        i = i - winSize

    if i > 0:
        smoothedSig = np.concatenate((smoothedSig, windowedSmoothingProcess(x[length - i:], len(x[length - i:]), tol, minSD)))

    return smoothedSig

def windowedSmoothing(x, winSize, tol, minSD):
    smoothedSignal = windowedSmoothingProcess(x, winSize, tol, minSD)
    return gaussian_filter1d(smoothedSignal, sigma=5, mode='reflect')

def gauss(x, sigmaD):
    return gaussian_filter1d(x, sigma=sigmaD, mode='reflect')

def savgolSmoothing (x, winL, ord, sigmaD)  :
    smooth = savgol_filter(x,winL,ord)
    return gaussian_filter1d(smooth, sigma=sigmaD, mode='reflect')

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

# works with PLETH
def divideMaximums(x, localMax):
    valuesMax = np.zeros((len(localMax), 1))
    for i in range(len(localMax)):
        valuesMax[i] = x[localMax[i]]
    
    meanM = np.mean(valuesMax)
    absMax = []
    locMax = []
    
    for value, index in zip(valuesMax, localMax):
        if value > meanM:
            absMax.append(value)
        else:
            locMax.append(value)
    
    absMax = np.array(absMax)
    locMax = np.array(locMax)
    
    return np.vstack((absMax, locMax))

def excludeMaximums2(x, locMax):
    valuesMax = []
    indexMax = []
    meanM = np.mean(x)
    for i in range(len(locMax)):
        if x[locMax[i]]>meanM:
            valuesMax.append(x[locMax[i]])
            indexMax.append(locMax[i])
    #print("VALMAX")
    #print(valuesMax)
    #print("IndMax")
    #print(indexMax)
    #print("ppp")
    #print(np.hstack((valuesMax,indexMax)))
    #print("ooo")
    #print(np.vstack((valuesMax,indexMax))[0,:])
    valuesMax = np.array(valuesMax)
    indexMax = np.array(indexMax)
    return valuesMax,indexMax

def excludeMinimums2(x,indexMax,locMin):
    valuesMin = []
    indexMin = []
    conflict = []
    toDel = [0]
    print(locMin)
    print(len(indexMax))
    print(indexMax)
    j=0
    for i in range(len(indexMax)-1):
        minVal = 999
        minInd = -1
        found = False
        final = False
        indM = locMin[j]
        while indM<indexMax[i]:
            j+=1
            indM = locMin[j]
        while indM>indexMax[i] and indM<indexMax[i+1] and (not final):
            found = True
            if minVal>x[indM]:
                    minVal = x[indM]
                    minInd = indM
            j += 1
            if j == len(locMin):
                final = True
            else:
                indM = locMin[j]
        if found:
                valuesMin.append(minVal)
                indexMin.append(minInd)
        else:
            toDel.append(i)
            conflict.append(indexMax[i])
    valuesMin = np.array(valuesMin)
    indexMin = np.array(indexMin)
    conflict = np.array(conflict)
    toDel = np.array(toDel)
    indexMax = np.delete(indexMax,toDel) 
    print(len(valuesMin)+len(conflict)-len(indexMax))
    print(conflict[0])
    print(valuesMin)
    print(len(valuesMin))
    print(indexMax)
    print(len(indexMax))
    #print(indexMin)
    return valuesMin,indexMin, conflict, toDel

def excludeMaximums(x, locMax):
    valuesMax = []
    indexMax = []
    meanM = np.mean(x)
    for i in range(len(locMax)):
        if x[locMax[i]]>meanM:
            valuesMax.append(x[locMax[i]])
            indexMax.append(locMax[i])
    #print("VALMAX")
    #print(valuesMax)
    #print("IndMax")
    #print(indexMax)
    #print("ppp")
    #print(np.hstack((valuesMax,indexMax)))
    #print("ooo")
    #print(np.vstack((valuesMax,indexMax))[0,:])
    valuesMax = np.array(valuesMax)
    indexMax = np.array(indexMax)
    return np.vstack((valuesMax,indexMax))


#CAN BE OPTIMISED!!!
def excludeMinimums(x,indexMax,locMin):
    valuesMin = []
    indexMin = []
    for i in range(len(indexMax)-1):
        minVal = 999
        minInd = -1
        found = False
        for j in range(len(locMin)):
            if locMin[j]>indexMax[i] and locMin[j]<indexMax[i+1]:
                found = True
                if minVal>x[locMin[j]]:
                    minVal = x[locMin[j]]
                    minInd = locMin[j]
        if found:
                valuesMin.append(minVal)
                indexMin.append(minInd)    
    valuesMin = np.array(valuesMin)
    indexMin = np.array(indexMin)
    #print(indexMin)
    return np.vstack((valuesMin,indexMin))


#def divideMaximums(x,localMax):
#    valuesMax = np.zeros((len(localMax),1))
#    for i in range(len(localMax)):
#        valuesMax[i] = x[localMax[i]]
#    meanM = np.mean(valuesMax)
#    absMax = np.array([])
#    locMax = np.array([])
#    for i in range(len(localMax)):
#        if valuesMax[i]>meanM:
#            absMax = np.append(absMax,localMax[i])
#        else:
#            locMax = np.append(locMax,localMax[i])
#    return np.vstack((absMax,locMax))

# works with PLETH
def divideMinimums(x, localMin):
    valuesMin = np.zeros((len(localMin), 1))
    for i in range(len(localMin)):
        valuesMin[i] = x[localMin[i]]
    
    meanM = np.mean(valuesMin)
    absMin = []
    locMin = []
    
    for value, index in zip(valuesMin, localMin):
        if value < meanM:
            absMin.append(value)
        else:
            locMin.append(value)
    
    absMin = np.array(absMin)#
    locMin = np.array(locMin)
    
    return np.vstack((absMin, locMin))

#def divideMinimums(x,localMin):
    valuesMin = np.zeros((len(localMin),1))
    for i in range(len(localMin)):
        valuesMin[i] = x[localMin[i]]
    meanM = np.mean(valuesMin)
    absMin = np.array([])
    locMin = np.array([])
    for i in range(len(localMin)):
        if valuesMin[i]<meanM:
            absMin = np.append(absMin,localMin[i])
        else:
            locMin = np.append(locMin,localMin[i])
    return np.vstack((absMin,locMin))

def cutData2D(x,y,discrim):
    cutVectX = []
    cutVectY = []
    meanX = np.mean(x)
    meanY = np.mean(y)
    stdX = np.std(x)
    stdY = np.std(y)
    for i in range(len(x)):
        if abs(x[i]-meanX)<stdX*discrim and abs(y[i]-meanY)<stdY*discrim:
            cutVectX.append(x[i])
            cutVectY.append(y[i])
    cutVectX = np.array(cutVectX)
    cutVectY = np.array(cutVectY)
    return cutVectX, cutVectY

def cutData2D2(x,y,discrim):
    cutVectX = []
    cutVectY = []
    toCut = []
    meanX = np.mean(x)
    meanY = np.mean(y)
    stdX = np.std(x)
    stdY = np.std(y)
    for i in range(len(x)):
        if abs(x[i]-meanX)<stdX*discrim and abs(y[i]-meanY)<stdY*discrim:
            cutVectX.append(x[i])
            cutVectY.append(y[i])
        else:
            toCut.append(i)
    cutVectX = np.array(cutVectX)
    cutVectY = np.array(cutVectY)
    toCut = np.array(toCut)
    return cutVectX, cutVectY, toCut

def comparePilot(pilot, x, lengthPilot):
    xx = np.linspace(0,lengthPilot,lengthPilot)
    x = x-np.mean(x)
    xxx = np.linspace(0,lengthPilot,len(x))
    xInter = CubicHermiteSpline(xxx,x, finiteDiffDiscrete(x))
    #Diff = (pilot-xInter)**2
    xxxx = np.linspace(0,lengthPilot,1000)
    # plt.figure()
    # plt.plot(xxxx,xInter(xxxx))
    # plt.plot(xxxx,pilot(xxxx))
    # plt.title("INTERP")
    # plt.show(block = False)
    # Int = 0
    # plt.figure()
    pilotCorr = np.array(pilot(xxxx))
    xCorr = np.array(xInter(xxxx))
    corr = np.correlate(xCorr,pilotCorr)
    return abs(corr[0])

