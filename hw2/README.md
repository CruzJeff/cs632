# CS632 Assignment 2 


 1. The PickleScript is used to extract the Cifar10 dataset using the pickle library, and to split it into 	the training and validation set. It also formats the images to be 32 x 32 so that it can be accepted 	by our neural network in the train.py script.

2. train.py loads the pickled training and validation sets and stores them into X train and Y train, and then X test and Y test respectively. Then the neural network's architecture is defined using the keras sequential model and maxpooling. The model is then trained using our training data, and it is saved for use in predict.py
	
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
            
The following is the summary of the model:

	model.summary()
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 30, 30, 32)        896       
    _________________________________________________________________
    activation_1 (Activation)    (None, 30, 30, 32)        0         
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 15, 15, 32)        0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 13, 13, 32)        9248      
    _________________________________________________________________
    activation_2 (Activation)    (None, 13, 13, 32)        0         
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 6, 6, 32)          0         
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 4, 4, 64)          18496     
    _________________________________________________________________
    activation_3 (Activation)    (None, 4, 4, 64)          0         
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 2, 2, 64)          0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 256)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 64)                16448     
    _________________________________________________________________
    activation_4 (Activation)    (None, 64)                0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 64)                0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 65        
    _________________________________________________________________
    activation_5 (Activation)    (None, 1)                 0         
    =================================================================
    Total params: 45,153
    Trainable params: 45,153
    Non-trainable params: 0


3. predict.py is a script takes that 1 input (a file containing data we want to make predictions on). It then loads the trained CNN model from train.py and makes predictions on the input data. For the purposes of testing, the model was trained using the entire training set, and tested using the validation set. The model showed a 93% accuracy on the training data, but a 71% accuracy on the testing data. This tells us that our model may be showing signs of overfitting. To improve this we could gather more testing data, or perhaps use a stronger model.
    
