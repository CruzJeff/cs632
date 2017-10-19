# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 06:26:14 2017

@author: User
"""

""" This code demonstrates reading the test data and writing 
predictions to an output file.
It should be run from the command line, with one argument:
$ python predict_starter.py [test_file]
where test_file is a .npy file with an identical format to those 
produced by extract_cats_dogs.py for training and validation.
(To test this script, you can use one of those).
This script will create an output file in the same directory 
where it's run, called "predictions.txt".
"""

import sys
import numpy as np
from keras.models import load_model
import time
model = load_model('my_model') #Load the trained CNN from Part 1

#Load Argument
try:
    sys.argv[1]
except IndexError:
    print("No arguments given. Please give a .npy file")
else:
    TEST_FILE = sys.argv[1]


data = np.load(TEST_FILE).item() #Unpack .npy file


images = data["images"] #Get Images 


ids = data["labels"] #Get Label for each testing image


OUT_FILE = "predictions.txt" #Name of OutPut file for predictions


#Make a prediction on each image and save output
out = open(OUT_FILE, "w")  
prediction = model.predict(images)
print("Writing predictions to file")
out.write("Labels" + " | " + "Predictions" + "\n")
for i, image in enumerate(images):

  image_id = ids[i]
  line = str(image_id) + "          " + str(prediction[i]) + "\n"
  out.write(line)
out.close() 

#Calculate errors
print("Calculating errors")
time.sleep(2)
error = 0
for x in range(2000):
    if ids[x] != round(prediction[x][0]):
        error = error + 1
print("Number of misclassifications: ",error)

