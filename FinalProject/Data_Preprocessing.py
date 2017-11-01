# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 06:48:32 2017

@author: User
"""

# Importing libraries
import numpy as np
import pandas as pd
import os

#Get Path to datasets
some_dir = input("What is the path to the directory that contains train and test.csv:  ")

#Change working directory to directory with files
os.chdir(some_dir)

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

# Importing the raw dataset
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
xtrain = train.shape[0]
xtest = test.shape[0]
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
for col in ('MSZoning','Electrical','KitchenQual','Exterior1st','Exterior2nd','SaleType'):
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
    
#Fill this missing value with Typical as said in description of dataset
all_data["Functional"] = all_data["Functional"].fillna("Typ")

#Drop Utilities, all except 3 have same value, highly unpredictive 
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

#Split into Train and Test
X_train = all_data[:xtrain]
X_test = all_data[xtrain:]


#Backward Elimination Feature Selection
import statsmodels.formula.api as sm
features = []
for x in range(1,219):
    features.append(x)
def backwards_elimination(features, X):
    X_opt = X.iloc[:, features]
    regressor_OLS = sm.OLS(endog=np.array(y_train), exog=X_opt).fit()
    p_values = regressor_OLS.pvalues
    for i,p_value in enumerate(p_values):
        if p_value>0.05:
            features.remove(features[i])  
            return backwards_elimination(features, X)
    return features
final_features = backwards_elimination(features, X_train)

#Get Columns to Remove
features_to_remove = []
for x in range(len(X_train.columns)):
    if x not in final_features:
        features_to_remove.append(x)

#Remove Non-Predictive Features
X_train = X_train.drop(X_train.columns[features_to_remove],axis=1)
X_test = X_test.drop(X_test.columns[features_to_remove],axis=1)

#Save Datasets 
X_train.to_csv("X_Train.csv", index=False)
y_train = pd.DataFrame(y_train.reshape((y_train.shape[0])))
y_train.to_csv("y_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
test_ID = pd.DataFrame(test_ID.reshape((test_ID.shape[0])))
test_ID.to_csv("test_ID.csv", index=False)