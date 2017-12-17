# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 07:29:33 2017

@author: User
"""
# Importing libraries
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from keras.wrappers.scikit_learn import KerasRegressor

# Importing the dataset
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
test_ID = pd.read_csv("test_ID.csv")['Id']

#Feature Scaling on Matrix of Features (These sets are only used for the DNN Regressor)
sc_X = StandardScaler()
X_train2 = sc_X.fit_transform(X_train)
X_test2 = sc_X.transform(X_test)

#Feature Scaling on Output Vector (This set is only used for the DNN Regressor)
sc_y = StandardScaler()
y_train2 = sc_y.fit_transform(y_train)
y_train = pd.read_csv("y_train.csv")["SalePrice"]
y_train2 = y_train2.reshape(1458)

#Load The Three Base Models
from sklearn.externals import joblib
XGB_Zenith = joblib.load('XGB_Zenith.pkl') #Loading saved XGBoost Regressor
Zenith_Ridge = joblib.load('Zenith_Ridge.pkl') #Loading saved Ridge Regressor
Zenith_Net = joblib.load('Zenith_Net.pkl') #Loading Elastic Net Regressor
Lasso_Zenith = joblib.load('lasso_zenith.pkl') #Loading Lasso Regressor
Light = joblib.load('light.pkl') #Loading LightBoost Regressor

#Creating the build function for the Deep Neural Network Regressor
def dnn():
    DNN = load_model('DNN_Regressor')
    return DNN

#Wrapping the Neural Network in a KerasRegressor to use with Ensemble Model

DNN_Regressor = KerasRegressor(build_fn=dnn, epochs=10, batch_size=5, verbose=0)


#Create Ensemble Model that uses the Average of base models
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y, X2, y2): #X2 and Y2 are the scaled training data
        self.models_ = []
        for x in self.models:
            self.models_.append(clone(x))
        
        # Train cloned base models
        for model in self.models_:
            if str(type(model)) == "<class 'keras.wrappers.scikit_learn.KerasRegressor'>":
            	#Separate branch because the DNN Regressor must be trained on the scaled dataset
                model.fit(X2,y2) #
            else:
                model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X, X2):
        predictions = []
        for model in self.models_:
            if str(type(model)) == "<class 'keras.wrappers.scikit_learn.KerasRegressor'>":
                predictions.append(sc_y.inverse_transform(model.predict(X2)))
            else:
                predictions.append(model.predict(X))
        predictions = np.column_stack(predictions)
        return np.mean(predictions, axis=1)   

#For the Average Model we used the 5th models that performed best on their own
AvgModel = AveragingModels(models=(XGB_Zenith,Lasso_Zenith,Zenith_Net,DNN_Regressor,Light)) 
AvgModel.fit(X_train,y_train, X_train2, y_train2)


#Creating a new ensemble model that uses the predictions of the base models as a new feature for the meta-model
from sklearn.model_selection import KFold
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    def fit(self, X, y, X2, y2):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
        	#Since the AVGModel contains the DNN, it must be handled separately
            if str(type(model)) != "<class '__main__.AveragingModels'>": 
                for train_index, holdout_index in kfold.split(X, y):
                    instance = clone(model)
                    self.base_models_[i].append(instance)
                    instance.fit(X[train_index], y[train_index])
                    y_pred = instance.predict(X[holdout_index])
                    out_of_fold_predictions[holdout_index, i] = y_pred
            else:
                for train_index, holdout_index in kfold.split(X, y):
                    instance = clone(model)
                    self.base_models_[i].append(instance)
                    instance.fit(X[train_index], y[train_index], X2, y2)
                    X_hold2 = sc_X.transform(X[holdout_index])
                    y_pred = instance.predict(X[holdout_index], X_hold2)
                    out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X, X2): 
        
        meta_features = np.column_stack([
        np.column_stack([model.predict(X) 
        if (str(type(model)) != "<class '__main__.AveragingModels'>") 
        else model.predict(X, X2)  #Special Consideration for the AVGModel's predict method
        for model in base_models]).mean(axis=1)
        for base_models in self.base_models_ ])
            
        print(meta_features)
        return self.meta_model_.predict(meta_features)

#For the stacked model, multiple combinations of base_models and meta_models were used until reaching this combination
stacked = StackingAveragedModels(base_models=(AvgModel,XGB_Zenith,Light),
                                 meta_model=Zenith_Net)

stacked.fit(X_train.values,y_train, X_train2, y_train2)


#Make Predictions
y_pred = stacked.predict(X_test.values, X_test2)
import math
for x in range (len(y_pred)):
    y_pred[x] = math.exp(y_pred[x]) 

#Output Predictions.csv
submission = pd.DataFrame({'SalePrice': y_pred.reshape((y_pred.shape[0])),'Id': test_ID})
submission.to_csv("./SalePrices.csv", index=False)
