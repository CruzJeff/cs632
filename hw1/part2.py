# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 19:19:07 2017

@author: User
"""

#Importing libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras.preprocessing import text, sequence
#Lists to store the instances and the labels
dataset = []
labels = []

def Euclid_distance(a,b):  #Calculate Distance using Numpy library
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)
    
#KNN Classifier from Part 1
class MyNearestNeighborClassifier():
    
    def __init__(self, n_neighbors=1):  #Constructor
        self.n_neighbors = n_neighbors
        
    def fit(self,X_train,y_train):  #Store training data
        self.X = X_train
        self.y = y_train
        if self.X is not list: #Convert Numpy Array to List if needed
            self.X = self.X.tolist()
        
    def predict(self, point):  #Make Predictions
            
        distances =[]
        neighbors = []
        nearest = {}
        for x in range(len(self.X)):  #Calculate Distance from test point to each instance in training data
            distance = Euclid_distance(point, self.X[x])
            distances.append( [self.X[x], distance] )
        distances.sort(key = lambda x: x[1]) #Sort distances in ascending order
        for x in range(self.n_neighbors):  #Add the K points with lowest distance to neighbors list
            neighbors.append(distances[x][0])
        for neighbor in neighbors:    #Count what each nearest neighbor is classified as
            response = self.y[self.X.index(neighbor)]
            if response in nearest:
                nearest[response] += 1
            else:
                nearest[response] = 1
        sorted_nearest = sorted(nearest.items(), key = lambda x: x[1], reverse=True) #Sort the counts in descending order
        
        return sorted_nearest[0][0] #Predict the majority class of the nearest neighbors
    
#Directory of the files
#some_dir = 'C:\\Users\\User\\Desktop\\misc-master\\misc-master\\spam_data'
some_dir = input("What is the directory with the data: ")
#Change working directory to directory with files
os.chdir(some_dir)

#Fill in the dataset with all of the instances
for filename in os.listdir(os.getcwd()):
    file = open(filename, 'r')
    dataset.append(file.read())
    file.close

#Split the dataset into training and test set
train_emails = dataset[:25]
test_emails= dataset[25:50]

#Fill in the labels dataset
file = open('labels.txt','r')
for word in file:
    labels.append(word[0])
    
#Split the labels into y_train and y_test
y_train = labels[:25]
y_test = labels[25:]

#Create bag of words
max_words = 250 #Assignment says to use 100, but I find that using 250 gave better results
tokenize = text.Tokenizer(num_words = max_words, char_level = False)
tokenize.fit_on_texts(train_emails)
x_train = tokenize.texts_to_matrix(train_emails)
x_test = tokenize.texts_to_matrix(test_emails)

#Create KNN Clasifier and fit it on training data
knn = MyNearestNeighborClassifier(3) #Tried using 1 to 10-Nearest Neighbors, 3 and 10 gave best and equal accuracies.
knn.fit(x_train,y_train)

#Make predictions on test data
predictions = []
for mail in x_test:
    predictions.append(knn.predict(mail))
print("My KNN Classifier made these predictions", predictions)
print("The actual labels are: ", y_test)

#Get accuracy of model
import sklearn.metrics
accuracy = sklearn.metrics.accuracy_score(y_test, predictions)
print("The accuracy of my KNN classifier is: ", accuracy)

#Find misclassifications
for x in range(25):
    if predictions[x] is not y_test[x]:
        print("Email number", x+25, "was misclassified")
