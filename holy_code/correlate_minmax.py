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

def corrMaxMain(x,winSize,sigma):
    if winSize>len(x):
        winSize=len(x)
    k = 1
    while k*winSize <= len(x):
        corrMax(x[(k-1)*winSize:k*winSize],sigma)
        k += 1
    if len(x)-(k-1)*winSize > 0:
        corrMax(x[(k-1)*winSize:len(x)],sigma)
    return 