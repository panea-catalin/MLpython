import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Load the dataset
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv"
df = pd.read_csv(url)

# Define feature set (X) and target variable (y)
X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ', 'retire', 'gender', 'reside']].values
y = df['custcat'].values

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# Normalize Data
scaler = preprocessing.StandardScaler()
X_train_norm = scaler.fit_transform(X_train.astype(float))
X_test_norm = scaler.transform(X_test.astype(float))

# K nearest neighbor (KNN)
k = 4  # Choose the number of neighbors
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train_norm, y_train)
yhat = neigh.predict(X_test_norm)

# Accuracy evaluation
train_accuracy = metrics.accuracy_score(y_train, neigh.predict(X_train_norm))
test_accuracy = metrics.accuracy_score(y_test, yhat)

print("Train set Accuracy: ", train_accuracy)
print("Test set Accuracy: ", test_accuracy)

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1, Ks):
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train_norm, y_train)
    yhat = neigh.predict(X_test_norm)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])

best_accuracy = mean_acc.max()
best_k = mean_acc.argmax() + 1
print("The best accuracy was with", best_accuracy, "with k =", best_k)

plt.plot(range(1, Ks), mean_acc, 'g')
plt.fill_between(range(1, Ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1, Ks), mean_acc - 3 * std_acc, mean_acc + 3 * std_acc, alpha=0.10, color="green")
plt.legend(('Accuracy', '+/- 1xstd', '+/- 3xstd'))
plt.ylabel('Accuracy')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()
