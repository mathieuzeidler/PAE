import numpy as np
import vitaldb as vd

def read_data(test, testObj):
    setD = []
    for f in test:
        setD.append((f, testObj.to_numpy(f, None)))
    return setD

#################################################################################################################
    
    ## Displaying the data and storing the values in the matrices

#################################################################################################################

#vitalDB data: 'Infinity/PLETH_SPO2', 'SNUADC/ART', 'SNUADC/PLETH'
#demo data: 'Infinity/PLETH_SPO2', 'Demo/ART', 'Demo/PLETH'
def process_data(setD, spo2name,artName,plethName):
    M_SPO2 = np.array([])
    M_ART = np.array([])
    M_PLETH = np.array([])

    for dataPair in setD:
        if dataPair[0] == spo2name:
            # Adding values to the M_SPO2 matrix without NaN
            values = dataPair[1][~np.isnan(dataPair[1])]
            if len(values) > 0:
                if M_SPO2.size == 0:
                    M_SPO2 = values
                else:
                    M_SPO2 = np.vstack((M_SPO2, values))

        elif dataPair[0] == artName:
            # Adding values to the M_ART matrix without NaN
            values = dataPair[1][~np.isnan(dataPair[1])]
            if len(values) > 0:
                if M_ART.size == 0:
                    M_ART = values
                else:
                    M_ART = np.vstack((M_ART, values))

        elif dataPair[0] == plethName:
            values = dataPair[1][~np.isnan(dataPair[1])]
            if len(values) > 0:
                if M_PLETH.size == 0:
                    M_PLETH = values
                else:
                    M_PLETH = np.vstack((M_PLETH, values))


    return M_SPO2, M_ART, M_PLETH

def stacker(directories):
    ARTFINAL = np.array([])
    PLETHFINAL = np.array([])
    for i in range(len(directories)):
        testObj = vd.read_vital(directories[i])
        test = testObj.get_track_names()
        setI = read_data(test,testObj)
        M_SPO2, M_ART, M_PLETH  = process_data(setI,'Infinity/PLETH_SPO2', 'Intellivue/ABP', 'Intellivue/PLETH')
        ARTFINAL = np.append(ARTFINAL,M_ART)
        PLETHFINAL = np.append(PLETHFINAL,M_PLETH)
    return ARTFINAL, PLETHFINAL

def display_matrices(M_SPO2, M_ART, M_PLETH):
    if M_SPO2.size > 0:
        print("Matrix M_SPO2:")
        print(M_SPO2)
        print(M_SPO2.size)

    if M_ART.size > 0:
        print("\nMatrix M_ART:")
        print(M_ART)
        print(M_ART.size)

    if M_PLETH.size > 0:
        print("\nMatrix M_PLETH:")
        print(M_PLETH)
        print(M_PLETH.size)
