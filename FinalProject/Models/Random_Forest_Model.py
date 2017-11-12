#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Different Heading because this part was made in Arch Linux in order to do gridsearch on multiple CPUs
"""
Created on Sat Oct 28 21:53:38 2017
@author: zeox101
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
from sklearn.ensemble import RandomForestRegressor
Random_Forest = RandomForestRegressor(n_estimators = 500, n_jobs=-1, random_state=42)
Random_Forest.fit(X_train, y_train)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = Random_Forest, X = X_train, y = y_train, cv = 10, 
                             scoring='neg_mean_squared_log_error', n_jobs=-1)
accuracies.mean()
accuracies.std()
print("Trained Basic_Random_Forest model with mean: ", accuracies.mean())
print("and stardard deviation of: ", accuracies.std())

# Applying Grid Search to find the best parameters
from sklearn.model_selection import GridSearchCV

#First tried all parameters, with n estimators from 100 to 1000 in intervals of 100,
#then 800 was picked.

#Grid search was rerun from 800 to 900 in intervals of 10 to find most optimal
#value for n_estimators, which lead to 890.

parameters = [{'n_estimators': [800, 810, 820, 830, 840, 850, 860, 870, 880, 890],
               'max_features': ['auto','sqrt','log2'],
               'max_depth' : [None,1,5,10,15,20,25,30],
               'min_samples_leaf' : [1,2,4,6,8,10,20],
               'min_samples_split' : [2,4] }]
    
#Ran for 1.5 hours on intel i7 6700HQ with 8 logical processors

grid_search = GridSearchCV(estimator = Random_Forest,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 10,
                           verbose = 1,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


#Create New Model Based on Results
Forest_Zenith = RandomForestRegressor(n_estimators=best_parameters['n_estimators'], #890
                                        max_features=best_parameters['max_features'], #auto
                                        max_depth = best_parameters['max_depth'], #15
                                    min_samples_leaf= best_parameters['min_samples_leaf'],
                                    min_samples_split=best_parameters['min_samples_split'], 
                                    )
                                        
Forest_Zenith.fit(X_train,y_train)

#Cross Validate New Model
accuracies = cross_val_score(estimator = Forest_Zenith, X = X_train, y = y_train, cv = 10, 
                             n_jobs=-1,scoring='neg_mean_squared_log_error')
accuracies.mean()
accuracies.std()
print("Trained Optimal_Random_Forest model with mean: ", accuracies.mean())
print("and standard deviation of: ", accuracies.std())


#Save Model
import pickle
regressor = Forest_Zenith
output = open('Forest_Zenith.pkl', 'wb')
pickle.dump(regressor, output)
output.close() 

#Make Predictions
y_pred = Forest_Zenith.predict(X_test)
import math
for x in range (len(y_pred)):
    y_pred[x] = math.exp(y_pred[x])

#Output Predictions.csv
submission = pd.DataFrame({'SalePrice': y_pred.reshape((y_pred.shape[0])),'Id': test_ID})
submission.to_csv("./SalePrices.csv", index=False)
