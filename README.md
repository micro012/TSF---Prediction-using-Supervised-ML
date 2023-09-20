# TSF---Prediction-using-Supervised-ML

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
data = pd.read_csv('/content/drive/MyDrive/GRIP internship/students_score.csv')
print(data)

#Plotting the 2-D chart
data.plot(x = 'Hours', y = 'Scores', style = 'o')
plt.title('Hours and Scores')
plt.xlabel('Hours studied')
plt.ylabel('Marks scored')
plt.show()

#Preparing the data
x = data.iloc[: , :1].values
y = data.iloc[: , -1].values
print(x)

#Importing the scikit learn library to split the data into test data and train data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size = 0.2, random_state = 0 )

#Training the model
#Importing linear regression from sklearn
from sklearn.linear_model import LinearRegression
regressor =  LinearRegression()
regressor.fit(x_train, y_train)\

#Plotting the regression line
line = regressor.coef_*x+regressor.intercept_
plt.scatter(x,y)
plt.plot(x, line, linewidth = 3 , color = 'red')
plt.show()
print(x_test)
y_pred = regressor.predict(x_test)
print(y_pred)
print(x_test)
result = regressor.predict([[9.25]])
print(result)

import math
from sklearn.metrics import mean_squared_error
error =math.sqrt(mean_squared_error(y_test, y_pred))
print(error)

