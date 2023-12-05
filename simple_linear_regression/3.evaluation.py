import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

# Load the processed 'cdf' DataFrame from the file saved by the first script
cdf = pd.read_csv('processed_data.csv')

# Train/Test Split (using 80% for training and 20% for testing)
msk = np.random.rand(len(cdf)) < 0.8
train = cdf[msk]  # Training dataset
test = cdf[~msk]  # Testing dataset

# Modeling: Creating a linear regression model using sklearn
train_x = np.asanyarray(train[['FUELCONSUMPTION_COMB']])  # Feature variable for training
train_y = np.asanyarray(train[['CO2EMISSIONS']])  # Target variable for training

test_x = np.asanyarray(test[['FUELCONSUMPTION_COMB']])  # Feature variable for testing
test_y = np.asanyarray(test[['CO2EMISSIONS']])  # Actual output for testing

regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)  # Fitting the model using the training data

# Predicting on test data
test_y_ = regr.predict(test_x)  # Predicted output for testing

# Model evaluation metrics
print("Mean Absolute Error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Mean Squared Error (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("Root Mean Squared Error (RMSE): %.2f" % np.sqrt(np.mean((test_y_ - test_y) ** 2)))
print("R-squared: %.2f" % r2_score(test_y, test_y_))
