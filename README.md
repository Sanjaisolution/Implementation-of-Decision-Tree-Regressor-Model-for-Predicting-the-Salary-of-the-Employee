# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries: Load necessary libraries for data manipulation, encoding, model training, and evaluation.
2.Load the dataset: Read the CSV file containing the salary data.
3.Explore the dataset:
Display the first few rows to understand the structure.
Check the data types and null values for each column.
Identify and handle any missing values.
Encode categorical data: Use LabelEncoder to transform categorical variables like "Position" into numerical values.
4.Feature selection: Select the independent variables (features) such as "Position" and "Level".
5.Select the target variable: Assign the dependent variable, "Salary", as the target.
6.Split the dataset: Divide the dataset into training and testing sets, ensuring a portion of the data is reserved for model evaluation.
7.Initialize and train the model: Use the Decision Tree Regressor to fit the model on the training data.
8.Make predictions: Use the trained model to predict salaries for the test dataset.
9.Evaluate the model:
10.Calculate the Mean Squared Error (MSE) to measure the average squared difference between actual and predicted values.
Calculate the R-squared score to determine how well the model explains the variability in the data.
Make new predictions: Use the model to predict the salary for a specific combination of "Position" and "Level" values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SANJAI.R
RegisterNumber:  212223040180
*/
```
```py
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
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
# data.head()
![Screenshot 2024-10-20 111549](https://github.com/user-attachments/assets/27d10b79-7d7b-42ac-aded-3b8a9ef83efb)

# data.info()
![Screenshot 2024-10-20 111554](https://github.com/user-attachments/assets/ebe94124-7166-48bf-8262-f3445fce6f5d)

# isnull() and sum()
![Screenshot 2024-10-20 111558](https://github.com/user-attachments/assets/7b59a876-d894-4ee6-9065-03a31309a67c)

# data.head() for salary
![Screenshot 2024-10-20 111603](https://github.com/user-attachments/assets/f1f5771b-689a-4bad-aca3-579dbf672c57)

# MSE value
![Screenshot 2024-10-20 111606](https://github.com/user-attachments/assets/49d66c74-aa54-478f-8c8a-7b660377b858)

# r2 value
![Screenshot 2024-10-20 111610](https://github.com/user-attachments/assets/52c94745-9dd8-4d36-8a6c-f34e87087c55)

# data prediction
![Screenshot 2024-10-20 111615](https://github.com/user-attachments/assets/021d79a6-2986-43f7-8a16-3d203b96b296)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
