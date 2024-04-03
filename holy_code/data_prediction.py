from sklearn.model_selection import train_test_split  # Importing function for splitting data into training and testing sets
from sklearn.linear_model import LinearRegression  # Importing the Linear Regression model
from sklearn import metrics  # Importing metrics to compute the prediction errors
import numpy as np  # Importing numpy for mathematical operations
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

# x the one to predict => ART and y => PLETH
def predict(x, y):  # Defining the prediction function that takes features (y) and target (x) as inputs
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)  # Splitting the data into training and testing sets

### Try several prediction models and compare them
    
    #model = LinearRegression() # Mean Absolute Error: 2.2465127537930596
                               # Mean Squared Error: 10.928526268996647
                               # Root Mean Squared Error: 3.3058321598345923

    #model = DecisionTreeRegressor() # Mean Absolute Error: 2.2465127537930596
                                    # Mean Squared Error: 10.928526268996647
                                    # Root Mean Squared Error: 3.3058321598345923 
    #model = SVR() # Mean Absolute Error: 1.9616585887477098
                  # Mean Squared Error: 8.271744394120887
                  # Root Mean Squared Error: 2.876064045552687
    #model = RandomForestRegressor() # good : Mean Absolute Error: 1.9883121185203616
                                    #        Mean Squared Error: 8.487491537348799
                                    #        Root Mean Squared Error: 2.9133299739900385


    model = GradientBoostingRegressor(n_estimators=100, max_depth=5) # good : Mean Absolute Error: 1.6022896128593709
                                        #        Mean Squared Error: 6.0157495883306895
                                        #        Root Mean Squared Error: 2.4527025070991977
    #model = KNeighborsRegressor() # good : Mean Absolute Error: 1.7536857020247725
                                  #        Mean Squared Error: 6.940321110475928
                                  #        Root Mean Squared Error: 2.634448919693819


    model.fit(X_train, y_train)  # Training the model using the training data


    predictions = model.predict(X_test)  # Making predictions using the trained model on the testing data

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))  # Printing the Mean Absolute Error of the predictions
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))  # Printing the Mean Squared Error of the predictions
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))  # Printing the Root Mean Squared Error of the predictions

    return predictions  # Returning the predictions