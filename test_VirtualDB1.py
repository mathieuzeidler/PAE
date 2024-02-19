import vitaldb as vd
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

DOWNLOAD_DIR = "dades/VitalDB_data/230602"
if not os.path.exists(DOWNLOAD_DIR):
    print("Creating new directory")
    os.mkdir(DOWNLOAD_DIR)
testObj = vd.read_vital("dades/VitalDB_data/230602/QUI12_230602_194231.vital")
test = testObj.get_track_names()
setD = list()
oo = testObj.to_numpy("MedibusX/PAMB_MBAR",None)
for f in test:
    setD.append((f,testObj.to_numpy(f,None)))
    
ppp = setD[0]
x = np.arange(0,len(ppp[1]))
plt.plot(x,ppp[1])