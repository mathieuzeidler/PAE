import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter

###############################################################################################################

    ## Local smoothing of data

###############################################################################################################
    

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
