# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 19:28:29 2017

@author: User
"""


 
# Assignment Part 1: K nearest neighbors classifier
# Should have constructor e.g clf = MyNearestNeighborClassifier(n_neighbors = 3)
# Should have fit function e.g clf.fit(X_train, y_train)
# should have a predict functino e.g clf.predict(X_test, y_test)
''' After you have implemented your classifier, calculate the accuracy on the test data using the
built in method as before, and compare it to the built in model. Ballpark, you should have similar
accuracy. You need not exactly replicate the results of the built-in classifier - but yours should
return correct predictions according to your distance metric.
'''
#import libraries
import numpy as np

#euclid distance function necessary for KNN algorithm
def Euclid_distance(a,b):  #Calculate Distance using Numpy library
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)

#KNN Classifier
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
    
#importing the dataset
from sklearn import datasets
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
#generating random permutations
np.random.seed(0)
indices = np.random.permutation(len(iris_X))

#splitting into training and testing data sets
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test  = iris_X[indices[-10:]]
iris_y_test  = iris_y[indices[-10:]]

#fitting the classifier on training set
clf = MyNearestNeighborClassifier(3)
clf.fit(iris_X_train, iris_y_train)
#making predictions on test set
predictions = []
for point in iris_X_test:
    predictions.append(clf.predict(point))
print("My KNN Classifier made these predictions", predictions)
print("Iris_Y_Test set is actually", iris_y_test)
#Output is [1, 2, 1, 0, 0, 0, 2, 1, 2, 0]

#iris_y_test is array([1, 1, 1, 0, 0, 0, 2, 1, 2, 0])

#Getting predictions from official KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train) 
print("The scikit learn KNN classifier made these predictions", knn.predict(iris_X_test))

#Output is  array([1, 2, 1, 0, 0, 0, 2, 1, 2, 0])
#The same output as MyNearestNeighborClassifier


#Getting the accuracy rating of my classifier
import sklearn.metrics
accuracy = sklearn.metrics.accuracy_score(iris_y_test, predictions)
print("The accuracy of my KNN classifier is: ", accuracy)
#Output is  0.90000000000000002

