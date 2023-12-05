import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Load the dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv"
df = pd.read_csv(url)

# Data Visualization and Analysis
print(df['custcat'].value_counts())

df.hist(column='income', bins=50)

# Define feature set (X) and target variable (y)
X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values
y = df['custcat'].values

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Normalize Data
scaler = preprocessing.StandardScaler()
X_train_norm = scaler.fit_transform(X_train.astype(float))
X_test_norm = scaler.transform(X_test.astype(float))
