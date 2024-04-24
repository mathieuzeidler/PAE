import vitaldb as vd
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

DOWNLOAD_DIR = "VitalDB_data/VitalDB_data/NewData"
if not os.path.exists(DOWNLOAD_DIR):
    print("Creating new directory")
    os.mkdir(DOWNLOAD_DIR)
#testObj = vd.read_vital("VitalDB_data/VitalDB_data/1.vital") 
testObj = vd.read_vital("VitalDB_data/VitalDB_data/230602/QUI12_230602_194231.vital")
test = testObj.get_track_names()
setD = list()
for f in test:
    setD.append((f,testObj.to_numpy(f,None)))

k = 1
for dataPair in setD:
    x = np.arange(0,len(dataPair[1]))
    print(dataPair[1])
    plt.figure(k)
    plt.stem(x,dataPair[1],linefmt='grey', markerfmt='')
    plt.xlabel("t")
    plt.ylabel(dataPair[0])
    plt.title(dataPair[0])
    #plt.plot(x,dataPair[1],'-o')
    k = k+1
plt.show()

vital1T = vd.read_vital("1.vital")
vital1TrackNames = vital1T.get_track_names()
vital1Tset = list()
vital1Tpandas
for f in test:
    setD.append((f,vital1T.to_numpy(f,None)))