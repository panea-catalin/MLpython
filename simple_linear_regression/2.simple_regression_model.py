import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Load the processed 'cdf' DataFrame from the file saved by the first script
cdf = pd.read_csv('processed_data.csv')

# Train/Test Split (using 80% for training and 20% for testing)
msk = np.random.rand(len(cdf)) < 0.8
train = cdf[msk]  # Training dataset
test = cdf[~msk]  # Testing dataset

# Visualizing the training data (Engine size vs CO2 Emissions)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Modeling: Creating a linear regression model using sklearn
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])  # Feature variable for training
train_y = np.asanyarray(train[['CO2EMISSIONS']])  # Target variable for training
regr.fit(train_x, train_y)  # Fitting the model using the training data

# Displaying the coefficients and intercept of the linear regression model
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

# Plotting the regression line over the training data
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
