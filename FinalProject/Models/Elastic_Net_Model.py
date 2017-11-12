# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 04:31:21 2017

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
        



#Create ElasticNet Regression Model
from sklearn.linear_model import ElasticNet
Elastic_Net = ElasticNet(random_state=42)
Elastic_Net.fit(X_train,y_train)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = Elastic_Net, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best parameters
from sklearn.model_selection import GridSearchCV

parameters = [{'alpha': [0.0005,0.005,0.05,0.5,1],
               'l1_ratio': [0.5,0.75,0.9,1],}]

grid_search = GridSearchCV(estimator = Elastic_Net,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 10,
                           verbose = 1,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#Create New Model Based on Results
Zenith_Net = ElasticNet(alpha=best_parameters['alpha'], #0.0005
                        l1_ratio=best_parameters['l1_ratio'], #0.5
                        random_state=42)

Zenith_Net.fit(X_train,y_train)

#Cross Validate New Model
accuracies = cross_val_score(estimator = Zenith_Net, X = X_train, y = y_train, cv = 10, 
                             n_jobs=-1,scoring='neg_mean_squared_error')
accuracies.mean()
accuracies.std()
print("Trained Optimal_Elastic_Net model with mean: ", accuracies.mean())
print("and standard deviation of: ", accuracies.std())

#Save Model
import pickle
regressor = Zenith_Net
output = open('Zenith_Net.pkl', 'wb')
pickle.dump(regressor, output)
output.close() 


#Make Predictions
y_pred = Zenith_Net.predict(X_test)
import math
for x in range (len(y_pred)):
    y_pred[x] = math.exp(y_pred[x]) 

#Output Predictions.csv
submission = pd.DataFrame({'SalePrice': y_pred.reshape((y_pred.shape[0])),'Id': test_ID})
submission.to_csv("./SalePrices.csv", index=False)















