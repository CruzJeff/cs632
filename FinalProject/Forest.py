#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 21:53:38 2017

@author: zeox101
"""


# Importing the libraries
import numpy as np
import pandas as pd
#Get Path to datasets
TRAIN_PATH = "/home/zeox101/Downloads/train.csv"
TEST_PATH = "/home/zeox101/Downloads/test.csv"


# Importing the dataset
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

#Remove Outliers
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)

#Take the log of SalePrice to match normal distribution and create y_train
train["SalePrice"] = np.log1p(train["SalePrice"])
y_train = train.iloc[:, 80]

#Save the 'Id' column
train_ID = train['Id']
test_ID = test['Id']
train.drop("Id", axis = 1, inplace = True)
test.drop("Id", axis = 1, inplace = True)

#Transform data
ntrain = train.shape[0]
ntest = test.shape[0]
all_data = pd.concat((train, test)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)

#Taking care of missing values

#Fill LotFrontage with Median
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))

#Fill these missing values with 0
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 
            'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath','MasVnrArea'):
    all_data[col] = all_data[col].fillna(0)
    
#FIll these missing values with the most common value
for col in ('MSZoning','Electrical','KitchenQual','Exterior1st','Exterior2nd',
            'SaleType'):
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
#Fill this missing value with Typical
all_data["Functional"] = all_data["Functional"].fillna("Typ")
#Drop Utilities, all except 3 have same value
all_data = all_data.drop(['Utilities'], axis=1)
#Replace all leftover missing values with None
all_data = all_data.fillna("None")

#Transforming "numerical" variables into categorical variables
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['OverallCond'] = all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)

#Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(all_data[c].values)) 
    all_data[c] = lbl.transform(list(all_data[c].values))

#Creating Dummy Variables
all_data = pd.get_dummies(all_data)
print(all_data.shape)

#Split into Train and Test
X_train = all_data[:ntrain]
X_test = all_data[ntrain:]
    


# Fitting the Regression Model to the dataset
from sklearn.ensemble import RandomForestRegressor
Random_Forest = RandomForestRegressor(n_estimators = 500, n_jobs=-1, random_state=42)
Random_Forest.fit(X_train, y_train)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = Random_Forest, X = X_train, y = y_train, cv = 10, n_jobs=-1)
accuracies.mean()
accuracies.std()

# Applying Grid Search to find the best parameters
from sklearn.model_selection import GridSearchCV
#First tried all parameters, with n estimators from 100 to 1000, then 
#800 was picked, and then narrowed down n_estimators itself to 890
parameters = [{'n_estimators': [800, 810, 820, 830, 840, 850, 860, 870, 880, 890],
               'max_features': ['auto','sqrt','log2'],
               'max_depth' : [None,1,5,10,15,20,25,30],
               'min_samples_leaf' : [1,2,4,6,8,10,20],
               'min_samples_split' : [2,4] }]
               
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
Optimal_Forest = RandomForestRegressor(n_estimators=best_parameters['n_estimators'],
                                        max_features=best_parameters['max_features'],
                                        max_depth = best_parameters['max_depth'],
                                    min_samples_leaf= best_parameters['min_samples_leaf'],
                                    min_samples_split=best_parameters['min_samples_split'], 
                                    )
                                        
                                                                    
                                
Optimal_Forest.fit(X_train,y_train)

#Cross Validate New Model
accuracies = cross_val_score(estimator = Optimal_Forest, X = X_train, y = y_train, cv = 10, 
                             n_jobs=-1,scoring='neg_mean_squared_log_error')
accuracies.mean()
accuracies.std()

#Get details of Model
print(Optimal_Forest)
'''RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=15,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0, min_impurity_split=None,
           min_samples_leaf=2, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=890, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)'''

#Save Model
import pickle
regressor = Optimal_Forest
output = open('Forest_Zenith.pkl', 'wb')
pickle.dump(regressor, output)
output.close() 

#Load MOdel
from sklearn.externals import joblib
Optimal_Forest = joblib.load('Optimal_Forest.pkl')

#Backward Elimination Feature Selection
import statsmodels.formula.api as sm
features = []
for x in range(1,219):
    features.append(x)
def backwards_eliminator(features, X):
    X_opt = X.iloc[:, features]
    regressor_OLS = sm.OLS(endog=np.array(y_train), exog=X_opt).fit()
    p_values = regressor_OLS.pvalues
    for i,p_value in enumerate(p_values):
        if p_value>0.05:
            features.remove(features[i])  
            return backwards_eliminator(features, X)
    return features, regressor_OLS
final_features, final_regressor = backwards_eliminator(features, X_train)

#Get Columns to Remove
features_to_remove = []
for x in range(len(X_train.columns)):
    if x not in final_features:
        features_to_remove.append(x)

#Remove Non-Predictive Features
X_train = X_train.drop(X_train.columns[features_to_remove],axis=1)
X_test = X_test.drop(X_test.columns[features_to_remove],axis=1)

# Make Predictions
y_pred2 = Optimal_Forest.predict(X_test)
import math
for x in range (1459):
    y_pred2[x] = math.exp(y_pred2[x])

# Output Predictions.csv
submission = pd.DataFrame({'SalePrice': y_pred2.reshape((y_pred2.shape[0])),'Id': test_ID})
submission.to_csv("./SalePrices.csv", index=False)
