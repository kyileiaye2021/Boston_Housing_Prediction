# Boston Housing Prediction

Using pandas, numpy, scikit-learn, scipy, matplotlib, and seaborn libraries, a **Multiple Linear Regression model** (MLR) was trained on [The Boston Housing Dataset](https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html) to predict the Median value of owner-occupied homes in $1000's.

Steps to build a MLR model:
1. Loading the dataset
2. Cleaning dataset (Checking missing value | Ruling out outliers in each column)
3. Preparing dataset for the model (Dataset splitting into training and testing data | feature scaling)
4. Training the model
5. Evaluating the model (Mean Absolute Error, Mean Squared Error | Root Mean Squared Error | r<sup>2</sup>)
6. Predicting the mdev of the input features


### Key Takeaways:
1. X_train and X_test: 2D pandas DataFrames
2. y_train and y_test: 1D pandas Series
3. y_pred is a 1D numpy array
4. Z-scores is number of standard deviation the data point is away from the mean value of its column in the dataset
5. Z-scores value is used for finding outliers in the dataset
6. Thereshold is usually 3 (If Z-scores > thereshold, the datapoint is considered an outlier)
7. Feature Scaling entures that all features contribute equally to the model
   
