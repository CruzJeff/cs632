# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 18:17:09 2017

@author: User
"""

# Importing libraries
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Importing the dataset
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
test_ID = pd.read_csv("test_ID.csv")['Id']

#Feature Scaling
#In theory this should not have an affect on the performance of a Deep Neural Net, but in practice it has shown to cause a slight boost in accuracy
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
y_train = y_train.reshape(1458)

#Splitting into training and validation sets
X_val = X_train[1166:]
X_train = X_train[:1166]
y_val = y_train[1166:]
y_train = y_train[:1166]


#Creating the model
model = Sequential()
model.add(Dense(119, input_dim=119, kernel_initializer='normal', activation='relu'))
model.add(Dense(60, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')

#Fitting and training the model
model.fit(X_train,y_train,validation_data=(X_val,y_val),
          epochs=10,batch_size=5,
          verbose=1)

#Saving model for later use
model.save('DNN_Regressor')

#Make Predictions
y_pred = model.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)
import math
for x in range (len(y_pred)):
    y_pred[x] = math.exp(y_pred[x]) 

#Output Predictions.csv
submission = pd.DataFrame({'SalePrice': y_pred.reshape((y_pred.shape[0])),'Id': test_ID})
submission.to_csv("./SalePrices.csv", index=False)
