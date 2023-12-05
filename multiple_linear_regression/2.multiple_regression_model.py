import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model

# Load the dataset
path = "FuelConsumption.csv"
df = pd.read_csv(path)

# Select features and target variable
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Split the data into train and test sets
train, test = train_test_split(cdf, test_size=0.2, random_state=42)

# Train the multiple regression model
regr = linear_model.LinearRegression()
x_train = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x_train, y_train)

# Test the model
x_test = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y_test = np.asanyarray(test[['CO2EMISSIONS']])
y_hat = regr.predict(x_test)

# Calculate evaluation metrics
print("Mean Squared Error (MSE): %.2f" % np.mean((y_hat - y_test) ** 2))
print('Variance score: %.2f' % regr.score(x_test, y_test))
