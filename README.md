# Implementation-of-Multivariate-Linear-Regression-Model-for-Sales-Prediction

## AIM:
To write a program to implement the multivariate linear regression model for sales prediction.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the multivariate linear regression model for sales prediction.
Developed by: Sai Eswar Kandukuri
RegisterNumber: 212221240020
*/
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('Advertising.csv')
df.head()
df.describe()
df.isnull().sum()
x = df[['TV','Radio','Newspaper']]
y = df["Sales"]
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=101)
from sklearn.linear_model import LinearRegression
l = LinearRegression()
l.fit(x_train,y_train)
y_pred = l.predict(x_test)
print("Regression slope: ",l.coef_[0])
print("Regression intercept:",l.intercept_)
from sklearn import metrics
MSE = metrics.mean_squared_error(y_test,y_pred)
print("MSE is {}".format(MSE))
r2 = metrics.r2_score(y_test,y_pred)
print("R squared error is {} ".format(r2))
l.predict([[150.3,240.5,234.5]])
```

## Output:
<img width="942" alt="Screenshot 2022-04-04 at 9 09 55 PM" src="https://user-images.githubusercontent.com/93427011/161581736-88b5a042-6761-4a95-861e-5f04747cf636.png">

## Result:
Thus the program to implement the multivariate linear regression model for sales prediction is written and verified using python programming.
