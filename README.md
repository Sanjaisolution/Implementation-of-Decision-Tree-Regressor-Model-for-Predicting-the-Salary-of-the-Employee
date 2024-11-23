# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1:Start the program.

Step 2:Import the required libraries.

Step 3:Upload the csv file and read the dataset.

Step 4:Check for any null values using the isnull() function.

Step 5:From sklearn.tree inport DecisionTreeRegressor.

Step 6:Import metrics and calculate the Mean squared error.

Step 7:Apply metrics to the dataset, and predict the output.

Step 8:End the prgoram.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: R.sanjai
RegisterNumber: 212223040180
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
mse
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
data.info() & data.isnull().sum():

![Screenshot 2024-10-07 110143](https://github.com/user-attachments/assets/aad12498-dbe2-4694-94ef-bef7eaedbba4)

data.head():

![Screenshot 2024-10-07 110149](https://github.com/user-attachments/assets/9211ae96-c336-403c-84c8-d79f4afa8c80)

MSE Value:

![Screenshot 2024-10-07 110156](https://github.com/user-attachments/assets/f34f78c1-b01b-43e7-987c-602a93544af3)

r2 Value:

![Screenshot 2024-10-07 110201](https://github.com/user-attachments/assets/e85e9490-477b-462a-ae7d-57a85d77d0ab)

Data Prediction:

![Screenshot 2024-10-07 110207](https://github.com/user-attachments/assets/a57d08b4-f002-4139-8355-4ab9b164557d)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
