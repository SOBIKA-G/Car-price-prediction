import pandas as pd
import numpy as np

car_data = pd.read_csv('/content/car data.csv')

print(car_data.head())

print(car_data.isna().sum())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
car_data['Car_Name'] = le.fit_transform(car_data['Car_Name'])
car_data['Fuel_Type'] = le.fit_transform(car_data['Fuel_Type'])
car_data['Selling_type'] = le.fit_transform(car_data['Selling_type'])
car_data['Transmission'] = le.fit_transform(car_data['Transmission'])

x = car_data.drop('Selling_Price',axis = 1)
y = car_data['Selling_Price']

print(x.head())
print(y.head())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 32)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score
print("MEAN SQUARED ERROR : ",mean_squared_error(y_test,y_pred))
print("R squared score : ",r2_score(y_test,y_pred))