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

#Generate IDs if not already present
if "ids" in data:
    ids = data["ids"]
else:
    # generate some random ids
    ids = list(range(0,len(images)))

data["ids"] = ids

OUT_FILE = "predictions.txt" #Name of OutPut file for predictions


#Make a prediction on each image and save output
out = open(OUT_FILE, "w")  
prediction = model.predict(images)
print("Writing predictions to file")
out.write("ids" + " | " + "Predictions" + "\n")
for i, image in enumerate(images):

  image_id = ids[i]
  line = str(image_id) + "          " + str(prediction[i]) + "\n"
  out.write(line)
out.close() 
