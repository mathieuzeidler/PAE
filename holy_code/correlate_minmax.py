import numpy as np
import math_func as mf
import matplotlib.pyplot as plt

def corrMax(x,sigma):
    smoothed = mf.savgolSmoothing(x,min(100,len(x)/2),2,sigma)
    smoothedMax = mf.localMaxOP2(smoothed)
    smoothedMin = mf.localMinOP2(smoothed)
    absMaxSmoothed = mf.excludeMaximums(smoothed,smoothedMax[0])
    absMinSmoothed = mf.excludeMinimums(smoothed,absMaxSmoothed[1,:],smoothedMin[0])
    return np.array([np.mean(absMaxSmoothed[0,:]),np.mean(absMinSmoothed[0,:])])

def corrMaxMain(x,y,winSize,sigma):
    maximumsVectX = []
    minimumsVectX = []
    maximumsVectY = []
    minimumsVectY = []
    winSizex = winSize
    if winSize>len(x):
        winSizex=len(x)
    k = 1
    while k*winSizex <= len(x):
        v = corrMax(x[(k-1)*winSizex:k*winSizex],sigma)
        maximumsVectX.append(v[0])
        minimumsVectX.append(v[1])
        k += 1
    if len(x)-(k-1)*winSizex > 0:
        v = corrMax(x[(k-1)*winSizex:len(x)],sigma)
        maximumsVectX.append(v[0])
        minimumsVectX.append(v[1])
    winSizey = winSize
    if winSize>len(y):
        winSizey=len(y)
    k = 1
    while k*winSizey <= len(y):
        v = corrMax(y[(k-1)*winSizey:k*winSizey],sigma)
        maximumsVectY.append(v[0])
        minimumsVectY.append(v[1])
        k += 1
    if len(y)-(k-1)*winSizey > 0:
        v = corrMax(y[(k-1)*winSizey:len(y)],sigma)
        maximumsVectY.append(v[0])
        minimumsVectY.append(v[1])
    maximumsVectX = np.array(maximumsVectX)
    minimumsVectX = np.array(minimumsVectX)
    maximumsVectY = np.array(maximumsVectY)
    minimumsVectY = np.array(minimumsVectY)
    print("MAXX")
    print(maximumsVectX)
    return 