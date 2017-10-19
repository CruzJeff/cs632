# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 05:09:28 2017

@author: User
"""

import numpy as np

TRAIN_PATH = "train.npy"  #Pickled training data
TEST_PATH = "validation.npy" #Pickled testing data

def load(npy_file):  #Load the pickled data
  data = np.load(npy_file).item()
  return data['images'], data['labels']

x_train, y_train = load(TRAIN_PATH) #Training data
x_test, y_test = load(TEST_PATH) #Testing data

#Creating the CNN
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32,32,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit(x_train, y_train, epochs=100, batch_size=80, verbose=True)

model.save('my_model')
