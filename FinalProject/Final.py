# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 07:29:33 2017
#Code for Stacking Model from https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard
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
        


#Load The Three Base Models
from sklearn.externals import joblib
XGB_Zenith = joblib.load('XGB_Zenith.pkl')
Zenith_Ridge = joblib.load('Zenith_Ridge.pkl')
Zenith_Net = joblib.load('Zenith_Net.pkl')
Lasso_Zenith = joblib.load('lasso_zenith.pkl')
Zenith_Forest = joblib.load('Zenith_Forest.pkl')
XGB_Zenith.fit(X_train,y_train)
Zenith_Ridge.fit(X_train,y_train)
Zenith_Net.fit(X_train,y_train)
Lasso_Zenith.fit(X_train,y_train)
Zenith_Forest.fit(X_train,y_train)

#Create Model
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)   

Chimera = AveragingModels(models=(XGB_Zenith,Lasso_Zenith))
Chimera.fit(X_train,y_train)

from sklearn.model_selection import KFold
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

stacked = StackingAveragedModels(base_models=(Chimera,XGB_Zenith),
                                 meta_model=Zenith_Net)
stacked.fit(X_train.values,y_train)

#Make Predictions
y_pred = stacked.predict(X_test.values)
import math
for x in range (len(y_pred)):
    y_pred[x] = math.exp(y_pred[x]) 

#Output Predictions.csv
submission = pd.DataFrame({'SalePrice': y_pred.reshape((y_pred.shape[0])),'Id': test_ID})
submission.to_csv("./SalePrices.csv", index=False)
