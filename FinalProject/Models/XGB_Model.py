# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 01:26:51 2017

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
        

#Create XGBoost Regressor
from xgboost import XGBRegressor

XGB = XGBRegressor(seed=42)
XGB.fit(X_train,y_train)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = XGB, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
print("Trained Basic_XGB_Regressor model with mean: ", accuracies.mean())
print("and stardard deviation of: ", accuracies.std())

# Applying Grid Search to find the best parameters
from sklearn.model_selection import GridSearchCV

parameters = [{'max_depth': [2,3,4],
               'learning_rate': [0.05,0.1],
               'n_estimators' : [300,500,1000,1500,2000],
               'min_child_weight' : [1,1.5,2],
               'reg_lambda' : [0.5,0.75,1],
               'reg_alpha' : [0,0.5], 
               'subsample' : [0.5,1] }]

grid_search = GridSearchCV(estimator = XGB,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 10,
                           verbose = 1,
                           n_jobs = -1)

#Ran for 2.5 Hours on Intel i7 6700HQ with 8 logical processors
grid_search = grid_search.fit(X_train, y_train)

best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


#Create New Model Based on Results
XGB_Zenith = XGBRegressor(n_estimators=best_parameters['n_estimators'],
                          learning_rate=best_parameters['learning_rate'],
                          max_depth = best_parameters['max_depth'],
                          min_child_weight= best_parameters['min_child_weight'],
                          reg_lambda=best_parameters['reg_lambda'], 
                          reg_alpha = best_parameters['reg_alpha'],
                          subsample = best_parameters['subsample'], )
                                    
                                    
                                        
XGB_Zenith.fit(X_train,y_train)


#Cross Validate New Model
accuracies = cross_val_score(estimator = XGB_Zenith, X = X_train, y = y_train, cv = 10, 
                             n_jobs=-1,scoring='neg_mean_squared_error')
accuracies.mean()
accuracies.std()
print("Optimized XGBRegressor model with mean: ", accuracies.mean())
print("and standard deviation of: ", accuracies.std())


#Save Model
import pickle
regressor = XGB_Zenith
output = open('XGB_Zenith.pkl', 'wb')
pickle.dump(regressor, output)
output.close() 


#Make Predictions
y_pred = XGB_Zenith.predict(X_test)
import math
for x in range (len(y_pred)):
    y_pred[x] = math.exp(y_pred[x]) 

#Output Predictions.csv
submission = pd.DataFrame({'SalePrice': y_pred.reshape((y_pred.shape[0])),'Id': test_ID})
submission.to_csv("./SalePrices.csv", index=False)














