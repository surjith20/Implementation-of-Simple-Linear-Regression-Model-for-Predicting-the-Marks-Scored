# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Surjith D
RegisterNumber:  212223043006
*/
```
```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```

## Output:

### Head Values
![image](https://github.com/user-attachments/assets/77c527c1-c491-4738-a244-dee93321c2b5)


### Tail Values
![image](https://github.com/user-attachments/assets/79f6d751-971f-4da1-b198-66b4926c5987)

### Compare Dataset 
![image](https://github.com/user-attachments/assets/c3d1f4f0-a1a4-4f5a-8667-cfa6a62c43bb)

### Predication values of X and Y
![image](https://github.com/user-attachments/assets/d6ed6a22-53cc-4dd9-aa43-25253cb17a14)


### Training set
![image](https://github.com/user-attachments/assets/4b1707fc-9c42-4053-bb22-b66076c24644)

### Testing Set
![image](https://github.com/user-attachments/assets/e161ca91-06ba-45c5-874e-558861b18988)

### MSE,MAE and RMSE
![image](https://github.com/user-attachments/assets/6249d2c9-fb04-453f-91ab-dd4f4acf6f1a)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
