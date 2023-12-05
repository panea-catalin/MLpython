import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import jaccard_score, confusion_matrix, classification_report, log_loss
from sklearn import preprocessing

# Load the dataset from the provided URL
def load_data():
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
    data = pd.read_csv(url)
    return data

# Preprocess the dataset
def preprocess_data(data):
    data = data[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless', 'churn']]
    data['churn'] = data['churn'].astype('int')
    X = np.asarray(data[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
    y = np.asarray(data['churn'])
    X = preprocessing.StandardScaler().fit(X).transform(X)
    return X, y

# Train the Logistic Regression model
def train_model(X_train, y_train):
    LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
    return LR

# Evaluate the trained model
def evaluate_model(model, X_test, y_test):
    yhat = model.predict(X_test)
    yhat_prob = model.predict_proba(X_test)
    
    jaccard = jaccard_score(y_test, yhat, pos_label=0)
    confusion_mat = confusion_matrix(y_test, yhat, labels=[1,0])
    report = classification_report(y_test, yhat)
    log_loss_value = log_loss(y_test, yhat_prob)
    
    return jaccard, confusion_mat, report, log_loss_value

def main():
    # Load data
    data = load_data()
    
    # Preprocess data
    X, y = preprocess_data(data)
    
    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    
    # Train the Logistic Regression model
    LR_model = train_model(X_train, y_train)
    
    # Evaluate the trained model
    jaccard_index, confusion_mat, classification_report, log_loss_value = evaluate_model(LR_model, X_test, y_test)
    
    # Print evaluation results
    print("Jaccard Index:", jaccard_index)
    print("Confusion Matrix:\n", confusion_mat)
    print("Classification Report:\n", classification_report)
    print("Log Loss:", log_loss_value)

if __name__ == "__main__":
    main()
