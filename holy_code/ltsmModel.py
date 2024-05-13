import numpy as np
import tf_keras
import matplotlib.pyplot as plt
import seaborn as sns
from tf_keras import layers, Sequential
from sklearn.preprocessing import StandardScaler

# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import Dense, Dropout
# import pandas as pd
# from matplotlib import pyplot as plt
# from sklearn.preprocessing import StandardScaler
# import seaborn as sns



def LSTMmain(x,y):


    scaler = StandardScaler()
    toTransform = np.append(x,np.atleast_2d(y).T,axis=1)
    scaler = scaler.fit(toTransform)
    transformed = scaler.transform(toTransform)
    x_ft_scaled = transformed[:,:np.shape(x)[1]]
    y_ft_scaled = transformed[:,-1]
    print(np.max(np.abs(x_ft_scaled[:,-4]-y_ft_scaled)))
    trainX = []
    trainY = []

    n_future = 0   # Number of cycles we want to look into the future based on the past cycles.
    n_past = 14  # Number of past cycles we want to use to predict the future.

    #Reformat input data into a shape: (n_samples x timesteps x n_features)
    #In my example, my df_for_training_scaled has a shape (12823, 5)
    #12823 refers to the number of data points and 5 refers to the columns (multi-variables).
    for i in range(n_past, len(x_ft_scaled) - n_future +1):
        trainX.append(x_ft_scaled[i - n_past:i, 0:x_ft_scaled.shape[1]])
        trainY.append(y_ft_scaled[i + n_future - 1:i + n_future])
    
    trainX, trainY = np.array(trainX), np.array(trainY)
    print('trainX shape == {}.'.format(trainX.shape))
    print('trainY shape == {}.'.format(trainY.shape))

    model = Sequential()
    
    model.add(layers.LSTM(128, activation='relu', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
    model.add(layers.LSTM(64, activation='relu', return_sequences=False))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(trainY.shape[1]))
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    history = model.fit(trainX, trainY, epochs=20, batch_size=16, validation_split=0.1, verbose=1)
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()

    n_days_for_prediction=32600  #let us predict past 15 days
    prediction = model.predict(trainX[-n_days_for_prediction:])
    plt.figure()
    plt.plot(prediction, '-r')
    plt.plot(y_ft_scaled, '-g')
    plt.plot(x_ft_scaled[:,0])
    #plt.plot(x_ft_scaled[:,1])
    #plt.plot(x_ft_scaled[:,2])
    plt.show()
    prediction_copies = np.repeat(prediction, toTransform.shape[1], axis=-1)
    y_pred_future = scaler.inverse_transform(prediction_copies)[:,-1]
    print(scaler.inverse_transform(prediction_copies))

    time = np.arange(len(x_ft_scaled))
    time2 = np.arange(len(y_pred_future))+len(time)
    plt.figure()
    plt.plot(y, '-g')
    plt.plot(y_pred_future, '-r')
    plt.show(block = False)

    plt.figure()
    plt.plot(np.abs(y_pred_future-y[-(n_days_for_prediction):]))
    plt.show()


maxvectPLETH_PRED = np.load("maxvertpleth.npy")
maxvectART = np.load("maxvertART.npy")
minvectPLETH_PRED = np.load("minvertpleth.npy")
minvectART = np.load("minvertart.npy")

LSTMmain(maxvectPLETH_PRED,maxvectART)
