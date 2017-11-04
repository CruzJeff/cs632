# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 13:37:19 2017

@author: User
"""



# Importing libraries
import numpy as np
import pandas as pd


# Importing the dataset
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")["SalePrice"]
X_test = pd.read_csv("X_test.csv")
test_ID = pd.read_csv("test_ID.csv")['Id']
        

#Create Basic Random_Forest Model
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(X_train,y_train)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = lasso, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best parameters
from sklearn.model_selection import GridSearchCV

parameters = [{'alpha': [0.0005,0.005,0.05,0.5,1]}]
               

grid_search = GridSearchCV(estimator = lasso,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 10,
                           verbose = 1,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#Create optimized model
lasso_zenith = Lasso(alpha=best_parameters['alpha'])
lasso_zenith.fit(X_train,y_train)

#Save Model
import pickle
regressor = lasso_zenith
output = open('lasso_Zenith.pkl', 'wb')
pickle.dump(regressor, output)
output.close() 

#Make Predictions
y_pred = lasso_zenith.predict(X_test)
import math
for x in range (len(y_pred)):
    y_pred[x] = math.exp(y_pred[x]) 

#Output Predictions.csv
submission = pd.DataFrame({'SalePrice': y_pred.reshape((y_pred.shape[0])),'Id': test_ID})
submission.to_csv("./SalePrices.csv", index=False)
