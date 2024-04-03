from sklearn.model_selection import train_test_split  # Importing function for splitting data into training and testing sets
from sklearn.linear_model import LinearRegression  # Importing the Linear Regression model
from sklearn import metrics  # Importing metrics to compute the prediction errors
import numpy as np  # Importing numpy for mathematical operations

# x the one to predict => ART and y => PLETH
def predict(x, y):  # Defining the prediction function that takes features (y) and target (x) as inputs
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)  # Splitting the data into training and testing sets

    model = LinearRegression()  # Creating a Linear Regression model instance
    model.fit(X_train, y_train)  # Training the model using the training data

    predictions = model.predict(X_test)  # Making predictions using the trained model on the testing data

    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))  # Printing the Mean Absolute Error of the predictions
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))  # Printing the Mean Squared Error of the predictions
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))  # Printing the Root Mean Squared Error of the predictions

    return predictions  # Returning the predictions