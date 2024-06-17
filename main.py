#Boston Housing Prediction

#Loading dataset
#Checking Missing Values
#Ruling out outliers
#Training the Multiple Linear Regression Model
#Evaluating the Model
'''
Key Takeaways
1. X_train and X_test: 2D pandas DataFrames
2. y_train and y_test: 1D pandas Series
3. y_pred is a 1D numpy array
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats #for z scores

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#load the dataset
data = pd.read_csv('BostonHousing.csv')

#------------Treating the missing value--------------
#finding the num of missing values in each column
num_of_missing_val = data.isna().sum()
print("Checking Missing Value Number")
print("==============================")
print(num_of_missing_val)
print()

'''
#Don't need to do this part because the dataset has no missing values
#Filling the cells of the columns that have missing values
na_columns = []
data[na_columns] = data[na_columns].fillna(data.mean()) #filling na cells with mean vals of dataset
'''

#------------------Checking Correlation--------------------
#visualizing correlation of each column relation with heatmap
corr_matrix = data.corr()
plt.figure(figsize=(12,8))
plt.title("Correlation Between Each Columns")
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="crest") 
plt.show()

#correlation of each column with the dependent variable
corr_matrix_medv = data.corrwith(data["medv"])
target_val_corr = np.abs(corr_matrix_medv)
print("Correlation between each independent feature to target feature")
print("==============================================================")
print("Features | Correlation with Medv")
print(target_val_corr.sort_values())
print()
'''
According to the heatmap and correlations, lstat is the most correlated with medv and chas is the least!
'''

#------------------Checking and Removing Outliers in the dataset---------------------
#checking outliers with Z-scores
#Z-scores is the number of standard deviation the data point is away from mean of its column in the dataset
z_scores = np.abs(stats.zscore(data)) #the z_scores of each cell is calculated and return dataframe in which each cell is the z_score
thereshold = 3 #commonly 3 | the datapoint is considered outlier if its z_score is above 3
outliers = np.where(z_scores > thereshold)
#print(outliers)

data_clean = data[(z_scores < thereshold).all(axis=1)] #keeping only the rows without outliers
print("***************************************Data Without Outliers************************************")
print(data_clean)
print(data_clean.columns)

#------------------------ Multiple Linear Regression Model ------------------------
#Splitting the dataset into independent and dependent variables
X = data_clean.drop(["medv"], axis=1)
y = data_clean["medv"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#Feature Scaling for adjusting all columns to have similar features
#Feature scaling ensures that all features contribute equally to the model

#Ensuring X_train and X_test are DataFrames before feature scaling
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

#Scaling X_train and X_test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#fitting the model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

#predicting the prices of X_test_scaled
y_pred = model.predict(X_test_scaled)


print("====== Actual House Prices vs. Predicted House Values ======")
for i in range(len(y_pred)):
    print(f"Actual Price: {y_test.iloc[i]}, Predicted Price: {y_pred[i]}")
print()


#Coefficients and Intercept
print("Coefficient")
coefficients = model.coef_
for feature, coef in zip(X_train.columns,coefficients):
    print(f"{feature}: {coef}")
print()

print(f"Intercept: {model.intercept_}")
print()

#----------------------------Plot The Result------------------------------
plt.scatter(y_test, y_pred)
#The line represents where the points would fall if the model's predictions were perfect (i.e., y_test equals y_pred for every point). 
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=2)
plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual House Prices vs. Predicted House Prices In Boston")
plt.show()

#Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")
r2 = r2_score(y_test, y_pred)
print(f"R-Squared: {r2}")
print()

# create a new DataFrame with the custom input values

custom_input = pd.DataFrame({
    'crim': [0.147],
    'zn':[2],
    'indus': [8.50],
    'chas': [0],
    'nox': [0.53],
    'rm': [6.728],
    'age': [79.5],
    'dis': [6],
    'rad': [5],
    'tax': [385],
    'ptratio':[20.9],
    'b':[395.0],
    'lstat':[9.42]
})
# scale the input values using the same scaling parameters as the training set
custom_input_scaled = scaler.transform(custom_input)

# make a prediction using the trained model
prediction = model.predict(custom_input_scaled)

# print the predicted value
print("Predicted value:", prediction[0])