# CS 632L Final Project Writeup
## Jeffrey Cruz
#
#
#
#### Project Description:
This project is based on the Kaggle Competition https://www.kaggle.com/c/house-prices-advanced-regression-techniques
In this competition the Ames House Dataset is provided (http://ww2.amstat.org/publications/jse/v19n3/decock.pdf) and used to create models that accurately predict the sale prices of the house. The data description.txt file provided gives details on exactly what features are given for each example in the dataset.

Here's a brief version of what you'll find in the data description file.
    
    SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
    MSSubClass: The building class
    MSZoning: The general zoning classification
    LotFrontage: Linear feet of street connected to property
    LotArea: Lot size in square feet
    Street: Type of road access
    Alley: Type of alley access
    LotShape: General shape of property
    LandContour: Flatness of the property
    Utilities: Type of utilities available
    LotConfig: Lot configuration
    LandSlope: Slope of property
    Neighborhood: Physical locations within Ames city limits
    Condition1: Proximity to main road or railroad
    Condition2: Proximity to main road or railroad (if a second is present)
    BldgType: Type of dwelling
    HouseStyle: Style of dwelling
    OverallQual: Overall material and finish quality
    OverallCond: Overall condition rating
    YearBuilt: Original construction date
    YearRemodAdd: Remodel date
    RoofStyle: Type of roof
    RoofMatl: Roof material
    Exterior1st: Exterior covering on house
    Exterior2nd: Exterior covering on house (if more than one material)
    MasVnrType: Masonry veneer type
    MasVnrArea: Masonry veneer area in square feet
    ExterQual: Exterior material quality
    ExterCond: Present condition of the material on the exterior
    Foundation: Type of foundation
    BsmtQual: Height of the basement
    BsmtCond: General condition of the basement
    BsmtExposure: Walkout or garden level basement walls
    BsmtFinType1: Quality of basement finished area
    BsmtFinSF1: Type 1 finished square feet
    BsmtFinType2: Quality of second finished area (if present)
    BsmtFinSF2: Type 2 finished square feet
    BsmtUnfSF: Unfinished square feet of basement area
    TotalBsmtSF: Total square feet of basement area
    Heating: Type of heating
    HeatingQC: Heating quality and condition
    CentralAir: Central air conditioning
    Electrical: Electrical system
    1stFlrSF: First Floor square feet
    2ndFlrSF: Second floor square feet
    LowQualFinSF: Low quality finished square feet (all floors)
    GrLivArea: Above grade (ground) living area square feet
    BsmtFullBath: Basement full bathrooms
    BsmtHalfBath: Basement half bathrooms
    FullBath: Full bathrooms above grade
    HalfBath: Half baths above grade
    Bedroom: Number of bedrooms above basement level
    Kitchen: Number of kitchens
    KitchenQual: Kitchen quality
    TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
    Functional: Home functionality rating
    Fireplaces: Number of fireplaces
    FireplaceQu: Fireplace quality
    GarageType: Garage location
    GarageYrBlt: Year garage was built
    GarageFinish: Interior finish of the garage
    GarageCars: Size of garage in car capacity
    GarageArea: Size of garage in square feet
    GarageQual: Garage quality
    GarageCond: Garage condition
    PavedDrive: Paved driveway
    WoodDeckSF: Wood deck area in square feet
    OpenPorchSF: Open porch area in square feet
    EnclosedPorch: Enclosed porch area in square feet
    3SsnPorch: Three season porch area in square feet
    ScreenPorch: Screen porch area in square feet
    PoolArea: Pool area in square feet
    PoolQC: Pool quality
    Fence: Fence quality
    MiscFeature: Miscellaneous feature not covered in other categories
    MiscVal: $Value of miscellaneous feature
    MoSold: Month Sold
    YrSold: Year Sold
    SaleType: Type of sale
    SaleCondition: Condition of sale 

### Step 1: Data Preprocessing
 The Data_Preprocessing.py file manipulates the data in ways that will help the models make more accurate predictions. We begin by removing outliers that we can see when making a scatterplot of the GrLivArea feature
 ![alt text]( https://image.prntscr.com/image/NRS312k1R4OhHnB3RuIWNA.png "GrLivArea Plot")
 
 We see that GrLivArea clearly has a positive linear relationship with GrLivArea, but we can see two outliers off to the right. This can be due to any number of reasons (A discounted sale, special circumstances, etc). So we begin by removing them
 
 Secondly, when we look at a graph of the output variable (SalePrice), we can see that it is a bit skewed left.
  ![alt text](https://image.prntscr.com/image/JKqgfVHCSDO3bkjn9DQ8lw.png "Seaborns Plot")
  
  Therefore we take the natural logarithm of this, so that the distribution will become identical to the normal distribution, which has been shown to help models make more accurate predictions.
  
  ![alt text](https://image.prntscr.com/image/qux_OJ9KT66MTf2UJ6qV4w.png "Seaborns Plot")
  
  The next and biggest part of the data preprocessing is handling missing values. We take special consideration for each feature when deciding how to handle missing values, based on what makes the most sense for that specific feature. Examples of methods used were, filling with median value of the column, removing the feature entirely if most of it was missing values, filling with the most common value, or replacing with "None".
  
  The fourth part of the data preprocessing is encoding the categorical variables so that they can be used in the learning models.
  
  The last part of the data preprocessing is feature selection. After encoding the categorical features and creating dummy variables for those features we end up with 219 total columns. It would help if we could trim down the number of columns, to possibly gain a boost in performance while not losing any accuracy. In general however, it is always better if you can obtain similar predictions with less information. It would mean less information would have to be gathered for future predictions. The method of feature selection I used is backwards elimination which removes all the low quality predictors until only the significant features remain. By doing this I was able to remove 100 columns, while still maintaining the same accuracy I had while using all 219 columns.
  
  
  
 ### Step 2: Model Building
 
The end-goal for this step is to create an ensemble model that will incorporate many machine learning models. This is to follow the "no free lunch" methodology. So I created many regressor models such as Random Forest, Elastic Net, Lasso, Ridge, XGBoost, LightBoost, and even a Deep Neural Network Regressor using Keras. The way I went about creating these models are all similar. I would first create a vanilla version of the model straight out of the box, and perform grid search to find the most optimal hyperparameters. I call these optimized versions the "Zenith" regressors after the word zenith which means "the time at which something is most powerful or successful." For the deep neural network however, I played around with different architectures and activation functions until I ended up with a neural network that performed just about as well as the other regressors. I evaluated all of these models using k-fold cross validation, and by also submitting their predictions of the test set on the kaggle competition. Each of these models can be found in the models folder. There are also pickled version of these models so that the scripts do not have to be run and grid search does not have to be re-done to get the most optimal models.

The next step was to create an ensemble model. In final.py, I started with a basic model that would just take the average of each model's predictions. The difficult part for this step was to find the right combination of base models to create the most accurate ensemble. Here are a few early results of the model building step (The score is Root Mean Logaramthic Squared Error).

 ![alt text](https://image.prntscr.com/image/TCjfxgDMQh2tfTBVVw0E_A.png "Early Results")
 
 The most accurate solo model ended up being the optimized Elastic Net, while the most accurate ensemble model was averaging XGBoost and Elastic Net.
 
 Next step was to make a more advanced ensembling method. The StackingAveragedModels regressor uses base models and a meta-model. The base models' predictions are taken and averaged, and then fed into the meta-model as a meta-feature in order to create more accurate predictions.
 
 ![alt text](https://image.prntscr.com/image/jtiUzttFSvikcyH8apicFg.png "StackingAveragedModels")
 
 The hardest part for this like with the previous ensemble model was finding the right combination of base models and meta model to get the most accurate predictions.
 
 ### Step 3: Evaluating Performance
 Here are each model's performance solo after hyperparameter optimization. Again, the metric shown is RMSLE (Root Mean Squared Logarithmic Error)
 
 Random Forest:
 ![alt text](https://image.prntscr.com/image/hgsYVTyDT4uobBJ8-jNKWg.png "RandomForest")
 
 XGBoost:
 ![alt text](https://image.prntscr.com/image/z_awxtdjR16YF1S9wHW90g.png "XGBoost")
 
 Elastic Net:
 ![alt text](https://image.prntscr.com/image/R5aSQ1wlSCeD2UljY9bR4Q.png "Elastic")
 
 Ridge:
 ![alt text](https://image.prntscr.com/image/Esas2iLSS3GOlXh2qLpzbg.png "Ridge")
 
 Lasso:
 ![alt text](https://image.prntscr.com/image/uIgvw3alTm252EhskuzGMw.png "Lasso")
 
 LightBoost:
 ![alt text](https://image.prntscr.com/image/0KIfzlodQg6EzxH0QjYy0Q.png "Light")
 
 Deep Neural Network:
 ![alt text](https://image.prntscr.com/image/SGOwTHqbTySZ_oWddOd5gg.png "DNN")
 
 The most succesful combination for the first averaging model:
![alt text]( https://image.prntscr.com/image/CjOftIpyRsi5M6efoG5x0g.png "StackingAveragedModels")

The most succesful combination for the second ensemble model:
![alt text](https://image.prntscr.com/image/MbsfRqcjR5SvZTDWamkmMQ.png "StackingAveragedModels")

This model using the Averaging Model (averaging XGB,Lasso,Elastic,DNN,and Light) alongside XGBoost as base models, while Elastic Net was used as the meta model. 

### Conclusion:
![alt text](https://image.prntscr.com/image/Iw54fTsjQOC938SOomiIRg.png "StackingAveragedModels")

To conclude, the final model was able to get into the top 13% of all entries for the kaggle competition. Possible next steps if we were trying to further improve our score would be to create more regressor models and see how they interact together in the ensemble models. Further data-preprocessing could also help boost the accuracy of our models. One thing that I did not do was feature extraction. Perhaps a more comprehensive look into the data could show that some features could be combined to create more features (Such as grouping Floor1Area and Floor2Area into a feature called TotalArea). 
