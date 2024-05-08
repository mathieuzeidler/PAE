import numpy as np
import math_func as mf
import math
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.interpolate import CubicHermiteSpline

def getMeans(x, ncicl):
    meanVect = []
    k = 0
    n1 = len(x)
    while n1-k>ncicl:
        mean = np.sum(x[k:k+ncicl])
        maxi = np.max(x[k:k+ncicl])
        mini = np.min(x[k:k+ncicl])
        mean = (mean - maxi - mini)/(ncicl-2)
        meanVect.append(mean)
        k += ncicl
    if n1-k>0:
        meanVect.append(np.mean(x[k:]))
    meanVect = np.array(meanVect)
    return meanVect

def corrMax2(smoothed):
    smoothedMax = mf.localMaxOP2(smoothed)
    smoothedMin = mf.localMinOP2(smoothed)
    print("CHECK 4-----------------")
    absMaxSmoothed, maxPos = mf.excludeMaximums2(smoothed,smoothedMax[0])
    print("CHECK 5-----------------")
    print(smoothedMin[0])
    print(maxPos)
    absMinSmoothed, minPos, critical, toDel = mf.excludeMinimums2(smoothed, maxPos,smoothedMin[0])
    absMaxSmoothed = np.delete(absMaxSmoothed,toDel)
    # plt.figure()
    # plt.plot(smoothed[0:5000000])
    # plt.vlines(critical*(critical<5000000), np.min(smoothed),np.max(smoothed),colors="red", alpha=0.5)
    # plt.vlines(minPos*(minPos<5000000), np.min(smoothed),np.max(smoothed),colors="orange", alpha=0.5)
    # plt.vlines(maxPos*(maxPos<5000000), np.min(smoothed),np.max(smoothed),colors="green", alpha=0.5)
    # plt.show()
    return absMaxSmoothed,absMinSmoothed, maxPos, minPos

def corrMaxMain2(x,y,sigma,ncicl,cutCorr):
    print("CHECK 3-----------------")
    maximumsVectXT, minimumsVectXT, maximumsPosX, minimumsPosX = corrMax2(x)
    maximumsVectYT, minimumsVectYT, maximumsPosY, minimumsPosY = corrMax2(y)

    pilot = np.load("PILOT.npy")
    pilot = pilot-np.mean(pilot)
    pilotInter = CubicHermiteSpline(np.arange(len(pilot)),pilot,mf.finiteDiffDiscrete(pilot))

    corrVect = []
    critCorr = []
    for i in range(len(minimumsPosX)-1):
        v = mf.comparePilot(pilotInter,x[minimumsPosX[i]:minimumsPosX[i+1]],len(pilot))
        corrVect.append(v)
        if(v<cutCorr):
            critCorr.append(i)
    critCorr = np.array(critCorr)
    corrVect = np.array(corrVect)

    maximumsPosX = np.delete(maximumsPosX,critCorr)
    minimumsPosX = np.delete(minimumsPosX,critCorr)
    maximumsPosY = np.delete(maximumsPosY,critCorr)
    minimumsPosY = np.delete(minimumsPosY,critCorr)

    maximumsVectXT = np.delete(maximumsVectXT,critCorr)
    minimumsVectXT = np.delete(minimumsVectXT,critCorr)
    maximumsVectYT = np.delete(maximumsVectYT,critCorr)
    minimumsVectYT = np.delete(minimumsVectYT,critCorr)

    print("CHECK 2-----------------")
    maximumsVectX = getMeans(maximumsVectXT,ncicl)
    maximumsVectY = getMeans(maximumsVectYT,ncicl)
    minimumsVectX = getMeans(minimumsVectXT,ncicl)
    minimumsVectY = getMeans(minimumsVectYT,ncicl)

    frecXT = mf.finiteDiffDiscrete(minimumsPosX)
    frecYT = mf.finiteDiffDiscrete(minimumsPosY)

    frecX = getMeans(frecXT,ncicl)
    frecY = getMeans(frecYT,ncicl)

    intXT = mf.integrate(x,minimumsPosX)
    intYT = mf.integrate(y,minimumsPosY)

    intX = getMeans(intXT,ncicl)
    intY = getMeans(intYT,ncicl)

    n1 = len(maximumsVectX)
    n2 = len(maximumsVectY)
    n3 = min(n1,n2)

    maximumsVectX = maximumsVectX[:n3]
    maximumsVectY = maximumsVectY[:n3]
    minimumsVectX = minimumsVectX[:n3]
    minimumsVectY = minimumsVectY[:n3]
    maximumsPosX = maximumsPosX[:n3]
    maximumsPosY = maximumsPosY[:n3]
    minimumsPosX = minimumsPosX[:n3]
    minimumsPosY = minimumsPosY[:n3]
    frecX = frecX[:n3]
    frecY = frecY[:n3]
    intX = intX[:n3]
    intY = intY[:n3]


    print("LENGTHS")
    print(len(minimumsPosX))
    print(len(maximumsPosX))
    print(len(minimumsPosY))
    print(len(maximumsPosY))
    print("CRITCORR")
    print(critCorr)

    # maximumsPosX = np.delete(maximumsPosX,critCorr)
    # minimumsPosX = np.delete(minimumsPosX,critCorr)
    # maximumsPosY = np.delete(maximumsPosY,critCorr)
    # minimumsPosY = np.delete(minimumsPosY,critCorr)

    # print("CHECK 99")
    # plt.figure()
    # plt.plot(corrVect)
    # plt.title("CORRELATION")
    # plt.show()

    print("CHECK 1-----------------")

    plt.figure()
    plt.scatter(maximumsVectX,maximumsVectY)
    plt.hlines(np.mean(maximumsVectY),-200,200, colors="red")
    plt.hlines(np.mean(maximumsVectY)-1.4*np.std(maximumsVectY),-200,200, colors="red")
    print("MMMMMM")
    print(np.mean(maximumsVectX))
    plt.vlines(np.mean(maximumsVectX),10,90, colors="orange")
    plt.vlines(np.mean(maximumsVectX)-1.4*np.std(maximumsVectX),10,90, colors="orange")
    plt.xlabel("Maximums PLETH")
    plt.ylabel("Maximums ART")
    plt.title("MAX SCATTERPLOT")
    plt.show(block=False)

    plt.figure()
    plt.scatter(minimumsVectX,minimumsVectY)
    plt.xlabel("Minimums PLETH")
    plt.ylabel("Minimums ART")
    plt.title("MIN SCATTERPLOT")
    plt.show(block=False)

    cutMaximumsVectX, cutMaximumsVectY, toCutMax = mf.cutData2D2(maximumsVectX,maximumsVectY,sigma)
    cutMinimumsVectX, cutMinimumsVectY, toCutMin = mf.cutData2D2(minimumsVectX,minimumsVectY,sigma)

    print("------------")
    print(len(minimumsPosX))
    print(len(minimumsPosY))
    print(len(frecX))
    print(len(frecY))
    print(len(intX))
    print(len(intY))

    if min(len(toCutMin),len(toCutMax))>0:
        maximumsPosX = np.delete(maximumsPosX,toCutMax)
        minimumsPosX = np.delete(minimumsPosX,toCutMin)
        maximumsPosY = np.delete(maximumsPosY,toCutMax)
        minimumsPosY = np.delete(minimumsPosY,toCutMin)
        if len(toCutMin) >= len(toCutMax):
            frecX = np.delete(frecX,toCutMin)
            frecY = np.delete(frecY,toCutMin)
            intX = np.delete(intX,toCutMin)
            intY = np.delete(intY,toCutMin)
        else:
            frecX = np.delete(frecX,toCutMax)
            frecY = np.delete(frecY,toCutMax)
            intX = np.delete(intX,toCutMax)
            intY = np.delete(intY,toCutMax)
    else:
        print("CUTMAX")
        print(len(toCutMax))
        print("CUTMIN")
        print(len(toCutMin))

    print("********")
    print(len(frecX))
    print(len(frecY))
    print(len(intX))
    print(len(intY))

    linMaxXY = linregress(cutMaximumsVectX, cutMaximumsVectY)
    plt.figure()
    plt.scatter(cutMaximumsVectX,cutMaximumsVectY)
    xLinSpace = np.linspace(-200,200,1000)
    plt.plot(xLinSpace, linMaxXY.intercept + linMaxXY.slope*xLinSpace, 'r', label='fitted line')
    plt.hlines(np.mean(maximumsVectY),-200,200, colors="red")
    plt.hlines(np.mean(maximumsVectY)-sigma*np.std(maximumsVectY),-200,200, colors="red")
    #print("MMMMMM")
    #print(np.mean(maximumsVectX))
    plt.vlines(np.mean(maximumsVectX),10,90, colors="orange")
    plt.vlines(np.mean(maximumsVectX)-sigma*np.std(maximumsVectX),10,90, colors="orange")
    plt.vlines(np.mean(maximumsVectX)+sigma*np.std(maximumsVectX),10,90, colors="orange")
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
    plt.hlines(np.mean(minimumsVectY)-sigma*np.std(minimumsVectY),-200,200, colors="red")
    #print("MMMMMM")
    #print(np.mean(maximumsVectX))
    plt.xlabel("MINIMUMS PLETH")
    plt.ylabel("MINIMUMS ART")
    plt.title("MIN SCATTERPLOT")
    plt.show(block=False)

    print("CORRELATION COEFF MAXMAX BEFORE CUT: ", np.corrcoef(maximumsVectX,maximumsVectY)[0,1])
    print("CORRELATION COEFF MINMIN BEFORE CUT: ", np.corrcoef(minimumsVectX,minimumsVectY)[0,1])
    print("CORRELATION COEFF MAXMAX: ", np.corrcoef(cutMaximumsVectX,cutMaximumsVectY)[0,1])
    print("CORRELATION COEFF MINMIN: ", np.corrcoef(cutMinimumsVectX,cutMinimumsVectY)[0,1])

    return cutMaximumsVectX, cutMaximumsVectY, cutMinimumsVectX, cutMinimumsVectY, maximumsPosX, maximumsPosY, minimumsPosX, minimumsPosY, frecX, frecY, intX, intY # X => PLETH, Y => ART

def corrMax(x,sigma, count):
    smoothed = []
    if len(x)<100:
        smoothed = mf.gauss(x,sigma)
    else:
        smoothed = mf.savgolSmoothing(x,100,2,sigma)
    smoothedMax = mf.localMaxOP2(smoothed)
    smoothedMin = mf.localMinOP2(smoothed)
    absMaxSmoothed, w = mf.excludeMaximums(smoothed,smoothedMax[0])
    absMinSmoothed, z = mf.excludeMinimums(smoothed,absMaxSmoothed[1,:],smoothedMin[0])
    return np.array([np.mean(absMaxSmoothed[0,:]),np.mean(absMinSmoothed[0,:])]), np.array([smoothedMax[0]])+count, np.array([smoothedMin[0]])+count

def corrMaxMain(x,y,winSize,sigma):
    maximumsVectX = []
    minimumsVectX = []
    maximumsVectY = []
    minimumsVectY = []
    minimumsPosX = []
    maximumsPosX = []
    minimumsPosY = []
    maximumsPosY = []
    winSizex = winSize
    if winSize>len(x):
        winSizex=len(x)
    k = 1
    count = 0
    while k*winSizex <= len(x):
        v, maxPos, minPos = corrMax(x[(k-1)*winSizex:k*winSizex],sigma,count)
        v[0] = 0 if math.isnan(v[0]) else v[0]
        v[1] = 0 if math.isnan(v[1]) else v[1]
        maximumsVectX.append(v[0])
        minimumsVectX.append(v[1])
        minimumsPosX = np.append(minimumsPosX,minPos)
        maximumsPosX = np.append(maximumsPosX,maxPos)
        count += k*winSizex
        k += 1
    if len(x)-(k-1)*winSizex > 0:
        v, maxPos, minPos = corrMax(x[(k-1)*winSizex:len(x)],sigma,count)
        v[0] = 0 if math.isnan(v[0]) else v[0]
        v[1] = 0 if math.isnan(v[1]) else v[1]
        maximumsVectX.append(v[0])
        minimumsVectX.append(v[1])
        minimumsPosX = np.append(minimumsPosX,minPos)
        maximumsPosX = np.append(maximumsPosX,maxPos)
    winSizey = winSize
    if winSize>len(y):
        winSizey=len(y)
    k = 1
    count = 0
    while k*winSizey <= len(y):
        v, maxPos, minPos = corrMax(y[(k-1)*winSizey:k*winSizey],sigma,count)
        v[0] = 0 if math.isnan(v[0]) else v[0]
        v[1] = 0 if math.isnan(v[1]) else v[1]
        maximumsVectY.append(v[0])
        minimumsVectY.append(v[1])
        minimumsPosY = np.append(minimumsPosY,minPos)
        maximumsPosY = np.append(maximumsPosY,maxPos)
        count += k*winSizey
        k += 1
    if len(y)-(k-1)*winSizey > 0:
        v, maxPos, minPos = corrMax(y[(k-1)*winSizey:len(y)],sigma,count)
        v[0] = 0 if math.isnan(v[0]) else v[0]
        v[1] = 0 if math.isnan(v[1]) else v[1]
        maximumsVectY.append(v[0])
        minimumsVectY.append(v[1])
        minimumsPosY = np.append(minimumsPosY,minPos)
        maximumsPosY = np.append(maximumsPosY,maxPos)
        count += k*winSizey
    maximumsVectX = np.array(maximumsVectX)
    minimumsVectX = np.array(minimumsVectX)
    maximumsVectY = np.array(maximumsVectY)
    minimumsVectY = np.array(minimumsVectY)

    ##MANIPULATE 


    print("MAXX")
    print(maximumsVectX)
    print(len(maximumsVectX))
    print(maximumsVectX[1000])
    print("MINX")
    print(minimumsVectX)
    print(len(minimumsVectX))
    print(minimumsVectX[1000])
    print("MAXY")
    print(maximumsVectY)
    print(len(maximumsVectY))
    print(maximumsVectY[1000])
    print(np.max(maximumsVectY))
    print("MINY")
    print(minimumsVectY)
    print(len(minimumsVectY))
    print(minimumsVectY[1000])
    
    plt.figure()
    plt.scatter(maximumsVectX,maximumsVectY)
    plt.hlines(np.mean(maximumsVectY),-200,200, colors="red")
    plt.hlines(np.mean(maximumsVectY)-1.4*np.std(maximumsVectY),-200,200, colors="red")
    print("MMMMMM")
    print(np.mean(maximumsVectX))
    plt.vlines(np.mean(maximumsVectX),10,90, colors="orange")
    plt.vlines(np.mean(maximumsVectX)-1.4*np.std(maximumsVectX),10,90, colors="orange")
    plt.xlabel("Maximums PLETH")
    plt.ylabel("Maximums ART")
    plt.title("MAX SCATTERPLOT")
    plt.show(block=False)

    plt.figure()
    plt.scatter(minimumsVectX,minimumsVectY)
    plt.xlabel("Minimums PLETH")
    plt.ylabel("Minimums ART")
    plt.title("MIN SCATTERPLOT")
    plt.show(block=False)

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

    print("CORRELATION COEFF MAXMAX BEFORE CUT: ", np.corrcoef(maximumsVectX,maximumsVectY)[0,1])
    print("CORRELATION COEFF MINMIN BEFORE CUT: ", np.corrcoef(minimumsVectX,minimumsVectY)[0,1])
    print("CORRELATION COEFF MAXMAX: ", np.corrcoef(cutMaximumsVectX,cutMaximumsVectY)[0,1])
    print("CORRELATION COEFF MINMIN: ", np.corrcoef(cutMinimumsVectX,cutMinimumsVectY)[0,1])

    return cutMaximumsVectX, cutMaximumsVectY, cutMinimumsVectX, cutMinimumsVectY, maximumsPosX, maximumsPosY, minimumsPosX, minimumsPosY # X => PLETH, Y => ART