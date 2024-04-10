import numpy as np
import math_func as mf
import math
import matplotlib.pyplot as plt
from scipy.stats import linregress

def corrMax(x,sigma):
    smoothed = []
    if len(x)<100:
        smoothed = mf.gauss(x,sigma)
    else:
        smoothed = mf.savgolSmoothing(x,100,2,sigma)
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
        v[0] = 0 if math.isnan(v[0]) else v[0]
        v[1] = 0 if math.isnan(v[1]) else v[1]
        maximumsVectX.append(v[0])
        minimumsVectX.append(v[1])
        k += 1
    if len(x)-(k-1)*winSizex > 0:
        v = corrMax(x[(k-1)*winSizex:len(x)],sigma)
        v[0] = 0 if math.isnan(v[0]) else v[0]
        v[1] = 0 if math.isnan(v[1]) else v[1]
        maximumsVectX.append(v[0])
        minimumsVectX.append(v[1])
    winSizey = winSize
    if winSize>len(y):
        winSizey=len(y)
    k = 1
    while k*winSizey <= len(y):
        v = corrMax(y[(k-1)*winSizey:k*winSizey],sigma)
        v[0] = 0 if math.isnan(v[0]) else v[0]
        v[1] = 0 if math.isnan(v[1]) else v[1]
        maximumsVectY.append(v[0])
        minimumsVectY.append(v[1])
        k += 1
    if len(y)-(k-1)*winSizey > 0:
        v = corrMax(y[(k-1)*winSizey:len(y)],sigma)
        v[0] = 0 if math.isnan(v[0]) else v[0]
        v[1] = 0 if math.isnan(v[1]) else v[1]
        maximumsVectY.append(v[0])
        minimumsVectY.append(v[1])
    maximumsVectX = np.array(maximumsVectX)
    minimumsVectX = np.array(minimumsVectX)
    maximumsVectY = np.array(maximumsVectY)
    minimumsVectY = np.array(minimumsVectY)

    ##MANIPULATE 


    #print("MAXX")
    #print(maximumsVectX)
    #print(len(maximumsVectX))
    #print(maximumsVectX[1000])
    #print("MINX")
    #print(minimumsVectX)
    #print(len(minimumsVectX))
    #print(minimumsVectX[1000])
    #print("MAXY")
    #print(maximumsVectY)
    #print(len(maximumsVectY))
    #print(maximumsVectY[1000])
    #print(np.max(maximumsVectY))
    #print("MINY")
    #print(minimumsVectY)
    #print(len(minimumsVectY))
    #print(minimumsVectY[1000])
    
    #plt.figure()
    #plt.scatter(maximumsVectX,maximumsVectY)
    #plt.hlines(np.mean(maximumsVectY),-200,200, colors="red")
    #plt.hlines(np.mean(maximumsVectY)-1.4*np.std(maximumsVectY),-200,200, colors="red")
    #print("MMMMMM")
    #print(np.mean(maximumsVectX))
    #plt.vlines(np.mean(maximumsVectX),10,90, colors="orange")
    #plt.vlines(np.mean(maximumsVectX)-1.4*np.std(maximumsVectX),10,90, colors="orange")
    #plt.xlabel("Maximums PLETH")
    #plt.ylabel("Maximums ART")
    #plt.title("MAX SCATTERPLOT")
    #plt.show(block=False)

    #plt.figure()
    #plt.scatter(minimumsVectX,minimumsVectY)
    #plt.xlabel("Minimums PLETH")
    #plt.ylabel("Minimums ART")
    #plt.title("MIN SCATTERPLOT")
    #plt.show(block=False)

    cutMaximumsVectX, cutMaximumsVectY = mf.cutData2D(maximumsVectX,maximumsVectY,3)
    cutMinimumsVectX, cutMinimumsVectY = mf.cutData2D(minimumsVectX,minimumsVectY,3)

    linMaxXY = linregress(cutMaximumsVectX, cutMaximumsVectY)
    plt.figure()
    plt.scatter(cutMaximumsVectX,cutMaximumsVectY)
    xLinSpace = np.linspace(-200,200,1000)
    plt.plot(xLinSpace, linMaxXY.intercept + linMaxXY.slope*xLinSpace, 'r', label='fitted line')
    plt.hlines(np.mean(maximumsVectY),-200,200, colors="red")
    plt.hlines(np.mean(maximumsVectY)-1.4*np.std(maximumsVectY),-200,200, colors="red")
    #print("MMMMMM")
    #print(np.mean(maximumsVectX))
    plt.vlines(np.mean(maximumsVectX),10,90, colors="orange")
    plt.vlines(np.mean(maximumsVectX)-1.4*np.std(maximumsVectX),10,90, colors="orange")
    plt.vlines(np.mean(maximumsVectX)+1.4*np.std(maximumsVectX),10,90, colors="orange")
    plt.xlabel("Maximums PLETH")
    plt.ylabel("Maximums ART")
    plt.title("MAX SCATTERPLOT")
    plt.show(block=False)

    linMinXY = linregress(cutMinimumsVectX, cutMinimumsVectY)
    plt.figure()
    plt.scatter(cutMinimumsVectX,cutMinimumsVectY)
    xLinSpace = np.linspace(-200,200,1000)
    plt.plot(xLinSpace, linMinXY.intercept + linMinXY.slope*xLinSpace, 'r', label='fitted line')
    plt.hlines(np.mean(minimumsVectY),-200,200, colors="red")
    plt.hlines(np.mean(minimumsVectY)-1.4*np.std(minimumsVectY),-200,200, colors="red")
    #print("MMMMMM")
    #print(np.mean(maximumsVectX))
    plt.xlabel("MINIMUMS PLETH")
    plt.ylabel("MINIMUMS ART")
    plt.title("MIN SCATTERPLOT")
    plt.show(block=False)

    #print("CORRELATION COEFF MAXMAX BEFORE CUT: ", np.corrcoef(maximumsVectX,maximumsVectY)[0,1])
    #print("CORRELATION COEFF MINMIN BEFORE CUT: ", np.corrcoef(minimumsVectX,minimumsVectY)[0,1])
    #print("CORRELATION COEFF MAXMAX: ", np.corrcoef(cutMaximumsVectX,cutMaximumsVectY)[0,1])
    #print("CORRELATION COEFF MINMIN: ", np.corrcoef(cutMinimumsVectX,cutMinimumsVectY)[0,1])

    return cutMaximumsVectX, cutMaximumsVectY, cutMinimumsVectX, cutMinimumsVectY # X => PLETH, Y => ART