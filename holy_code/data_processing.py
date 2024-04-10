import numpy as np

def read_data(test, testObj):
    setD = []
    for f in test:
        setD.append((f, testObj.to_numpy(f, None)))
    return setD

#################################################################################################################
    
    ## Displaying the data and storing the values in the matrices

#################################################################################################################


#vitalDB dataset names: spo2Name-> Infinity/PLETH_SPO2, artName-> SNUADC/ART, plethName->SNUADC/PLETH
#Demo data: names->Demo/x
def process_data(setD,spo2Name,artName,plethName):
    M_SPO2 = np.array([])
    M_ART = np.array([])
    M_PLETH = np.array([])

    for dataPair in setD:
        if dataPair[0] == spo2Name:
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
