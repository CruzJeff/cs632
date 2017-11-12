# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 02:47:51 2017

@author: User
"""

# Importing libraries
import pandas as pd


# Importing the dataset
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")["SalePrice"]
X_test = pd.read_csv("X_test.csv")
test_ID = pd.read_csv("test_ID.csv")['Id']
        


from lightgbm import LGBMRegressor
light = LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
                
light.fit(X_train,y_train)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = light, X = X_train, y = y_train, cv = 10, 
                             scoring='neg_mean_squared_error')
print("Mean: ", accuracies.mean())
print("Standard Deviation: ", accuracies.std())

#Save Model
import pickle
regressor = light
output = open('Light.pkl', 'wb')
pickle.dump(regressor, output)
output.close() 

#Make Predictions
y_pred = light.predict(X_test)
import math
for x in range (len(y_pred)):
    y_pred[x] = math.exp(y_pred[x]) 

#Output Predictions.csv
submission = pd.DataFrame({'SalePrice': y_pred.reshape((y_pred.shape[0])),'Id': test_ID})
submission.to_csv("./SalePrices.csv", index=False)
