from sklearn.model_selection import train_test_split  # Importing function for splitting data into training and testing sets
from sklearn.linear_model import LinearRegression  # Importing the Linear Regression model
from sklearn import metrics  # Importing metrics to compute the prediction errors
import numpy as np  # Importing numpy for mathematical operations
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

def usemodel(X_train,X_test,y_train,y_test,model,name):

    model.fit(X_train, y_train)  # Training the model using the training data
    predictions = model.predict(X_test)  # Making predictions using the trained model on the testing data
    print('                                  ')
    print("------Results of",name,":--------")
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))  # Printing the Mean Absolute Error of the predictions
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))  # Printing the Mean Squared Error of the predictions
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))  # Printing the Root Mean Squared Error of the predictions

    return predictions  # Returning the predictions

###############################################################################################################
######################################### Just MALAK COMMENTS #################################################
###############################################################################################################

#Correlations:
#cutMaximumsVectX, cutMaximumsVectY = corrMaxMain(M_ART,M_PLETH,2000,sigma)
#cutMaximumsVectX_array = np.array(cutMaximumsVectX).reshape(-1, 1)
# Linear regression model to predict the correlation between the ART and PLETH signals
#correlation_coefficient = np.corrcoef(cutMaximumsVectX, cutMaximumsVectY)[0, 1]
#print("Correlation coefficient:", correlation_coefficient)

# Polynomial regression model to predict the correlation between the ART and PLETH signals
#poly = PolynomialFeatures(degree=2)
#cutMaximumsVectX_poly = poly.fit_transform(cutMaximumsVectX_array)
#model = LinearRegression()
#model.fit(cutMaximumsVectX_poly, cutMaximumsVectY)
# y = 0.00000000e+00 + (-1.07488142e-02)*x + 1.78275323e-05*x^2 + 63.16531559476246
# R-squared score: 0.00467589165993243

# Tree regression model to predict the correlation between the ART and PLETH signals
#model = DecisionTreeRegressor(max_depth=10) # we can adjust the depth
#model.fit(cutMaximumsVectX_array, cutMaximumsVectY)
#print("Depth of the tree:", model.get_depth())
#Depth of the tree: 39 --> overfitting: the model learns the training data too well,
#                                       including its noise and outliers,
#                                       which can lead to poor performance on new, unseen data
#R-squared score: 1.0 --> The model is perfectly predicting the training data

#Depth of the tree: 10
#R-squared score: 0.32915647428623396

# Support Vector Regression (SVR) model to predict the correlation between the ART and PLETH signals
#model = SVR(kernel='rbf')
#model.fit(cutMaximumsVectX_array, cutMaximumsVectY)
#R-squared score: 0.004523067848973272

###############################################################################################################
###############################################################################################################
# Gradient Boosting Regression model to predict the correlation between the ART and PLETH signals
# model = GradientBoostingRegressor(n_estimators=100, max_depth=13)
# model.fit(cutMaximumsVectX_array, cutMaximumsVectY)
# R-squared score: 0.28817222879730764 for n_estimators=100, max_depth=3
# R-squared score: 0.4778262112518894 for n_estimators=100, max_depth=5
# R-squared score: 0.7249963779958533 for n_estimators=100, max_depth=8
# R-squared score: 0.9248403061986639 for n_estimators=100, max_depth=13
###############################################################################################################
###############################################################################################################

# Predict the values of cutMaximumsVectY based on cutMaximumsVectX_poly
#cutMaximumsVectY_pred = model.predict(cutMaximumsVectX_array)
# Calculate the R-squared score
#r2 = r2_score(cutMaximumsVectY, cutMaximumsVectY_pred)
#print("R-squared score:", r2)

# x the one to predict => ART and y => PLETH
def predict(x, y):  # Defining the prediction function that takes features (y) and target (x) as inputs
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)  # Splitting the data into training and testing sets
    model = LinearRegression() # Mean Absolute Error: 2.2465127537930596
                               # Mean Squared Error: 10.928526268996647
                               # Root Mean Squared Error: 3.3058321598345923
    predictionLinearR  = usemodel(X_train,X_test,y_train,y_test,model,"Linear Regression")
    model = DecisionTreeRegressor() # Mean Absolute Error: 2.2465127537930596
                                    # Mean Squared Error: 10.928526268996647
                                    # Root Mean Squared Error: 3.3058321598345923 
    predictionDecisionT  = usemodel(X_train,X_test,y_train,y_test,model,"Decision Tree Regressor")
    model = SVR() # Mean Absolute Error: 1.9616585887477098
                  # Mean Squared Error: 8.271744394120887
                  # Root Mean Squared Error: 2.876064045552687
    predictionSVR  = usemodel(X_train,X_test,y_train,y_test,model, "SVR")
    model = RandomForestRegressor() # good : Mean Absolute Error: 1.9883121185203616
                                    #        Mean Squared Error: 8.487491537348799
                                    #        Root Mean Squared Error: 2.9133299739900385
    predictionRandomF = usemodel(X_train,X_test,y_train,y_test,model, "Random Forest Regressor")

    model = GradientBoostingRegressor(n_estimators=100, max_depth=13) # good : Mean Absolute Error: 1.6022896128593709
                                        #        Mean Squared Error: 6.0157495883306895
                                        #        Root Mean Squared Error: 2.4527025070991977
    predictionGradientB = usemodel(X_train,X_test,y_train,y_test,model,"Gradient Boosting Regressor")
    model = KNeighborsRegressor() # good : Mean Absolute Error: 1.7536857020247725
                                  #        Mean Squared Error: 6.940321110475928
                                  #        Root Mean Squared Error: 2.634448919693819
    predictionKNeighbors = usemodel(X_train,X_test,y_train,y_test,model, "KNeaighbors Regressor")

    toReturn = {'LinearRegression':predictionLinearR,'DecissionTree':predictionDecisionT,'SVR':predictionSVR,'RandomForest':predictionRandomF,'GradientBoosting':predictionGradientB,'KNeighbors':predictionKNeighbors}

    return toReturn