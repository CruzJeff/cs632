1. In a Nearest Neighbor classifier, is it important that all features be on the same scale?
	Think: what would happen if one feature ranges between 0-1, and another ranges
	between 0-1000? If it is important that they are on the same scale, how could you
	achieve this?

Yes, it is important that all features be on the same scale. If all the features were not on the same scale, then the features that ranged between 0-1 would arbitrarily have a lesser distance than the ones that ranged from 0-1000, and this would mess up our classifier's predictions. They can be set to the same scale using Feature Scaling. For Example:

    ```Python
    	from sklearn.preprocessing import StandardScaler
	sc_X = StandardScaler()
	X_train = sc_X.fit_transform(X_train)
	X_test = sc_X.transform(X_test)
	sc_y = StandardScaler()
	y_train = sc_y.fit_transform(y_train
```
2. What is the difference between a numeric and categorical feature? How might you
	represent a categorical feature so your Nearest Neighbor classifier could work with it?
    
A numeric feature is a feature that is represented by a number like prices in dollars, height in inches, etc. A categorical feature is like a label, a feature that is not represented numerically like State, Height Description(Tall, Short), etc. We can use one hot encoding to let our nearest neighbor classifier work with categorical features. For example:

  ```Python
    	from sklearn.preprocessing import LabelEncoder, OneHotEncoder
	labelencoder_X = LabelEncoder()
	X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
	onehotencoder = OneHotEncoder(categorical_features = [3])
	X = onehotencoder.fit_transform(X).toarray()
```
3. What is the importance of testing data?
		
        The importance of testing data is to make sure that the model is not merely memorizing the training data and can make accurate predictions on new instances. The testing data, and possibly even a validation set, can be used to prevent overfitting on the training set.
        
        
        
4. What does “supervised” refer to in “supervised classification”?

	"Supervised" means that we know the labels of the data in our training set. Conversely, unupervised would mean that we don't know what the true classification of the data in our training set are, meaning the model would be learning blind. However with supervised learning, the model can learn to make predictions by learning from instances of data and its label.
    
  
5. If you were to include additional features for the Iris dataset, what would they be, and
why?

	A possible additional feature that could be useful is the hue of the petals. The Iris Viriginica is visibly a lighter shade of purple than the setosa and versicolor. So if we could somehow quantify the shade of colors of each instance, a metric that reflects a lighter shade could help predict an Iris Virginica.
