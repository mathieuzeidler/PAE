from sklearn.model_selection import train_test_split, RandomizedSearchCV  # Importing function for splitting data into training and testing sets
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression, ElasticNet, RidgeCV, RidgeCV, LassoCV  # Importing the Linear Regression model
from sklearn import metrics  # Importing metrics to compute the prediction errors
import numpy as np  # Importing numpy for mathematical operations
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from skopt.space import Integer, Real
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

def usemodel(X_train,X_test,y_train,y_test,model,name, full, X,Y):

    model.fit(X_train, y_train)  # Training the model using the training data
    yt = y_test
    if full:
        predictions = model.predict(X)  # Making predictions using the trained model on the testing data
        yt = Y
    else:
        predictions = model.predict(X_test)  # Making predictions using the trained model on the testing data
    print('                                  ')
    print("------Results of ",name,":--------")
    print('Mean Absolute Error:', metrics.mean_absolute_error(yt, predictions))  # Printing the Mean Absolute Error of the predictions
    print('Mean Squared Error:', metrics.mean_squared_error(yt, predictions))  # Printing the Mean Squared Error of the predictions
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(yt, predictions)))  # Printing the Root Mean Squared Error of the predictions
    print('Score:',metrics.r2_score(yt, predictions))
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
def predict(x, y, full, first):  # Defining the prediction function that takes features (y) and target (x) as inputs
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)  # Splitting the data into training and testing sets
    # X_TRAIN = np.array((6,))
    # Y_TRAIN = np.array((6,))
    # K = len(X_train[:,0])//6
    # for i in range(5):
    #     X_TRAIN[i] = X_train[i*K:(i+1)*K,:]
    #     Y_TRAIN[i] = y_train[i*K:(i+1)*K,:]
    # X_TRAIN[5] = X_train[i*5:,:]
    # Y_TRAIN[5] = X_train[i*5:,:]
    model = LinearRegression() # Mean Absolute Error: 2.2465127537930596
                               # Mean Squared Error: 10.928526268996647
                               # Root Mean Squared Error: 3.3058321598345923
    predictionLinearR  = usemodel(X_train,X_test,y_train,y_test,model,"Linear Regression", full, x,y)
    model = DecisionTreeRegressor() # Mean Absolute Error: 2.2465127537930596
                                    # Mean Squared Error: 10.928526268996647
                                    # Root Mean Squared Error: 3.3058321598345923 
    predictionDecisionT  = usemodel(X_train,X_test,y_train,y_test,model,"Decision Tree Regressor", full, x,y)
    model = Lasso(alpha=2) # Mean Absolute Error: 2.2465127537930596
                                    # Mean Squared Error: 10.928526268996647
                                    # Root Mean Squared Error: 3.3058321598345923 
    predictionLasso  = usemodel(X_train,X_test,y_train,y_test,model,"Lasso Regressor", full, x,y)
    model = SVR(C=0.1,epsilon=0.1005) # Mean Absolute Error: 1.9616585887477098
                  # Mean Squared Error: 8.271744394120887
                  # Root Mean Squared Error: 2.876064045552687
    predictionSVR  = usemodel(X_train,X_test,y_train,y_test,model, "SVR", full, x, y)
    #'bootstrap': False, 'max_depth': 60, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 700
    #bootstrap = False, max_depth =  60, max_features = 'sqrt', min_samples_leaf = 1, min_samples_split = 3, n_estimators = 700
    #bootstrap = False, max_depth =  50, max_features = 'sqrt', min_samples_leaf = 1, min_samples_split = 2, n_estimators = 800
    model = RandomForestRegressor(bootstrap = False, max_depth =  50, max_features = 'sqrt', min_samples_leaf = 1, min_samples_split = 2, n_estimators = 800) # good : Mean Absolute Error: 1.9883121185203616
                                    #        Mean Squared Error: 8.487491537348799
                                    #        Root Mean Squared Error: 2.9133299739900385
    predictionRandomF = usemodel(X_train,X_test,y_train,y_test,model, "Random Forest Regressor", full, x,y)

    model = HistGradientBoostingRegressor(loss='squared_error',early_stopping=False, max_depth=None, l2_regularization=0.1,learning_rate=0.1,max_iter=1200,min_samples_leaf=15) # good : Mean Absolute Error: 1.6022896128593709
                                        #        Mean Squared Error: 6.0157495883306895
                                        #        Root Mean Squared Error: 2.4527025070991977
    predictionGradientB = usemodel(X_train,X_test,y_train,y_test,model,"Gradient Boosting Regressor", full, x,y)
    model = KNeighborsRegressor() # good : Mean Absolute Error: 1.7536857020247725
                                  #        Mean Squared Error: 6.940321110475928
                                  #        Root Mean Squared Error: 2.634448919693819
    predictionKNeighbors = usemodel(X_train,X_test,y_train,y_test,model, "KNeaighbors Regressor", full, x,y)

    toReturn = {'LinearRegression':predictionLinearR,'DecissionTree':predictionDecisionT,'SVR':predictionSVR,'RandomForest':predictionRandomF,'GradientBoosting':predictionGradientB,'KNeighbors':predictionKNeighbors, 'Lasso':predictionLasso}

    return toReturn , y_test

def Stacking(model,train,y,test,n_fold, already, folds):
    if not already:
        folds=KFold(n_splits=n_fold,random_state=None)
    train_pred = []
    test_pred=np.empty((test.shape[0],1),float)
    for train_indices,val_indices in folds.split(train,y):
        x_train,x_val=train[train_indices],train[val_indices]
        y_train,y_val=y[train_indices],y[val_indices]
        model.fit(X=x_train,y=y_train)
        train_pred=np.append(train_pred,model.predict(x_val))
    model.fit(X=train,y=y)
    train_pred = np.atleast_2d(np.array(train_pred)).T
    test_pred= np.atleast_2d(np.array(model.predict(test))).T
    return test_pred,train_pred, folds

def predict2(x, y, finalmodel):  # Defining the prediction function that takes features (y) and target (x) as inputs
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)  # Splitting the data into training and testing sets
    model = LinearRegression() # Mean Absolute Error: 2.2465127537930596
                               # Mean Squared Error: 10.928526268996647
                               # Root Mean Squared Error: 3.3058321598345923
    test_pred1 ,train_pred1, folds =Stacking(model=model,n_fold=10,train=X_train,test=X_test,y=y_train, already=False, folds=None)
    model = DecisionTreeRegressor() # Mean Absolute Error: 2.2465127537930596
                                    # Mean Squared Error: 10.928526268996647
                                    # Root Mean Squared Error: 3.3058321598345923 
    test_pred2 ,train_pred2, xx=Stacking(model=model,n_fold=10,train=X_train,test=X_test,y=y_train,already=True, folds=folds)
    trEns = np.append(train_pred1, train_pred2, axis=1)
    teEns = np.append(test_pred1, test_pred2, axis=1)
    model = Lasso(alpha=0.85) # Mean Absolute Error: 2.2465127537930596
                                    # Mean Squared Error: 10.928526268996647
                                    # Root Mean Squared Error: 3.3058321598345923 
    test_pred2 ,train_pred2, xx=Stacking(model=model,n_fold=10,train=X_train,test=X_test,y=y_train,already=True, folds=folds)
    trEns = np.append(trEns, train_pred2, axis=1)
    teEns = np.append(teEns, test_pred2, axis=1)
    model = SVR(C=0.1,epsilon=0.1005) # Mean Absolute Error: 1.9616585887477098
                  # Mean Squared Error: 8.271744394120887
                  # Root Mean Squared Error: 2.876064045552687
    test_pred2 ,train_pred2, xx=Stacking(model=model,n_fold=10,train=X_train,test=X_test,y=y_train,already=True, folds=folds)
    trEns = np.append(trEns, train_pred2, axis=1)
    teEns = np.append(teEns, test_pred2, axis=1)
    model = RandomForestRegressor() # good : Mean Absolute Error: 1.9883121185203616
                                    #        Mean Squared Error: 8.487491537348799
                                    #        Root Mean Squared Error: 2.9133299739900385
    test_pred2 ,train_pred2, xx=Stacking(model=model,n_fold=10,train=X_train,test=X_test,y=y_train,already=True, folds=folds)
    trEns = np.append(trEns, train_pred2, axis=1)
    teEns = np.append(teEns, test_pred2, axis=1)

   
    model = HistGradientBoostingRegressor(loss='squared_error',early_stopping=False, max_depth=None) # good : Mean Absolute Error: 1.6022896128593709
                                        #        Mean Squared Error: 6.0157495883306895
                                        #        Root Mean Squared Error: 2.4527025070991977
    test_pred2 ,train_pred2, xx=Stacking(model=model,n_fold=10,train=X_train,test=X_test,y=y_train,already=True, folds=folds)
    trEns = np.append(trEns, train_pred2, axis=1)
    teEns = np.append(teEns, test_pred2, axis=1)
    model = KNeighborsRegressor() # good : Mean Absolute Error: 1.7536857020247725
                                  #        Mean Squared Error: 6.940321110475928
                                  #        Root Mean Squared Error: 2.634448919693819
    test_pred2 ,train_pred2, xx=Stacking(model=model,n_fold=10,train=X_train,test=X_test,y=y_train,already=True, folds=folds)
    trEns = np.append(trEns, train_pred2, axis=1)
    teEns = np.append(teEns, test_pred2, axis=1)
    
    finalmodel.fit(trEns,y_train)
    predictions = finalmodel.predict(teEns)

    return predictions , y_test, teEns, finalmodel

def predict3(x,y):
    #model = HistGradientBoostingRegressor(loss='squared_error',early_stopping=False, max_depth=None)
    model = GradientBoostingRegressor()
    params = {"learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 1],"max_depth": [ 3, 4, 5, 6, 8, 10, 12, 15], "alpha": [0.1, 0.2 , 0.3, 0.4, 0.8, 0.9 ], "min_impurity_decrease" : [0.0, 0.2, 0.4 , 0.7, 5 ]}
    searchParams = RandomizedSearchCV(model,param_distributions=params,n_iter=100,n_jobs=-1,cv=5, scoring='neg_mean_squared_error', verbose=1)
    predictions , y_test, teEns, finalmodel = predict2(x,y,searchParams)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))  # Printing the Mean Absolute Error of the predictions
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))  # Printing the Mean Squared Error of the predictions
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))  # Printing the Root Mean Squared Error of the predictions
    print(finalmodel.score(teEns,y_test))
    return {'RESULT':predictions}, y_test


def predict4(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)  # Splitting the data into training and testing sets
    estimators = [('linear', LinearRegression()), 
                  ('SVR', SVR(C=0.1,epsilon=0.1005)) ,
                  ('lasso', LassoCV(alphas=np.array([0.85,0.85,0.85,0.85,0.85]))), 
                  ('knr', KNeighborsRegressor(n_neighbors=20, metric='euclidean')),
                  ('RNDF', RandomForestRegressor(bootstrap = False, max_depth =  50, max_features = 'sqrt', min_samples_leaf = 1, min_samples_split = 2, n_estimators = 800)), 
                  ('HGDB', HistGradientBoostingRegressor(loss='squared_error',early_stopping=False, max_depth=None, l2_regularization=0.1,learning_rate=0.1,max_iter=1200,min_samples_leaf=15))]
    final_estimator = GradientBoostingRegressor(n_estimators=30, subsample=0.5, min_samples_leaf=25, max_features=1)
    #final_estimator = HistGradientBoostingRegressor(loss='squared_error',early_stopping=False, max_depth=None, l2_regularization=0.1,learning_rate=0.1,max_iter=1200,min_samples_leaf=15)
    regressor = StackingRegressor(estimators=estimators,final_estimator=final_estimator, n_jobs=-1)
    #regressor = MLPRegressor(max_iter=9999)
    regressor.fit(X_train,y_train)
    print(regressor.score(X_test,y_test))
    pred = regressor.predict(X_test)
    print("MAE:", metrics.mean_absolute_error(pred,y_test), sep=" ")
    print("MSE:", metrics.mean_squared_error(pred,y_test), sep=" ")
    return {'RESULT': regressor.predict(X_test)}, y_test

def scoringHGB(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)  # Splitting the data into training and testing sets
    param_grid = {
    'learning_rate': [.1, .12],
    'max_iter': [1000, 1200],
    'min_samples_leaf': [15, 20],
    'max_depth': [None, 10, 20],
    'l2_regularization': [0.1, 0.01, 0.001]
    }
    model = HistGradientBoostingRegressor()
    modelS = GridSearchCV(model, param_grid=param_grid,cv=10, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
    modelS.fit(X_train,y_train)
    print(modelS.best_params_)
    print(modelS.best_score_)

def scoringRF(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)  # Splitting the data into training and testing sets
    # param_grid= {
    #     'bootstrap': [True, False],
    #     'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    #     'max_features': ['auto', 'sqrt'],
    #     'min_samples_leaf': [1, 2, 4],
    #     'min_samples_split': [2, 5, 10],
    #     'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    # }
    param_grid= {
        'bootstrap': [True, False],
        'max_depth': [40, 50, 60],
        'max_features': ['sqrt'],
        'min_samples_leaf': [1, 2],
        'min_samples_split': [2, 3, 5],
        'n_estimators': [700, 800, 900]
    }
    model = RandomForestRegressor() 
    modelS = GridSearchCV(model, param_grid=param_grid,cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=4)
    modelS.fit(X_train,y_train)
    print(modelS.best_params_)
    print(modelS.best_score_)

def scoringLASSO(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)  # Splitting the data into training and testing sets
    # param_grid= {
    #     'bootstrap': [True, False],
    #     'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
    #     'max_features': ['auto', 'sqrt'],
    #     'min_samples_leaf': [1, 2, 4],
    #     'min_samples_split': [2, 5, 10],
    #     'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    # }
    param_grid = {
        'alpha' : np.arange(0.00, 1.0, 0.01)
    }
    model = Lasso() 
    modelS = GridSearchCV(model, param_grid=param_grid,cv=10, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    modelS.fit(X_train,y_train)
    print(modelS.best_params_)
    print(modelS.best_score_)

def scoringSVR(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)  # Splitting the data into training and testing sets
    param_grid = {
        'C' : np.arange(0.00, 1.0, 0.01),
        'epsilon' : np.arange(0.01,1,100)
    }
    model = SVR() 
    modelS = GridSearchCV(model, param_grid=param_grid,cv=10, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    modelS.fit(X_train,y_train)
    print(modelS.best_params_)
    print(modelS.best_score_)
    

# def predictLayered(x, y, n):
#     currentX = x
#     for i in range(n-1):
#         print("-------NEW LAYER-------")
#         predictions, ytest = predict(currentX,y,True)
#         for j in predictions.keys():
#             # print(predictions[j])
#             # print(np.shape(predictions[j]))
#             # print(np.shape(currentX))
#             toAdd = np.array(predictions[j])
#             # print(np.shape(toAdd))
#             currentX = np.append(currentX, np.atleast_2d(toAdd).T, axis=1)
#         print(np.shape(currentX))    
#     predictions, ytest = predict(currentX,y,False)
#     return predictions, ytest


def plotRes(dictX,ytest):
    print("KEYYYYY")
    for i in dictX.keys():
        plt.figure()
        plt.plot(dictX[i], 'r-')
        plt.plot(ytest,'b-', alpha=0.5)
        plt.title("Results of: " + i)
        plt.show(block=False)

        plt.figure()
        plt.plot(np.abs(dictX[i]-ytest))
        plt.title("Puntual Error of: " + i)
        plt.show(block=False)

    return 
        