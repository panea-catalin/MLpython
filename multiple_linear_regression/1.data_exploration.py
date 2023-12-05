import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np

# URL of the CSV file to download
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv'

# Destination file name to save the downloaded CSV file
file_name = 'FuelConsumption.csv'

# Downloading the CSV file using requests
response = requests.get(url)
if response.status_code == 200:
    with open(file_name, 'wb') as f:
        f.write(response.content)
    print(f"File '{file_name}' downloaded successfully.")
else:
    print("Failed to download the file.")

# Read the downloaded CSV file using pandas
df = pd.read_csv(file_name)

# Display the first few rows of the dataframe
print(df.head())

# Plot Emission values with respect to Engine size
plt.scatter(df['ENGINESIZE'], df['CO2EMISSIONS'], color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

# Split the dataset into train and test sets (80% training, 20% testing)
msk = np.random.rand(len(df)) < 0.8
train = df[msk]
test = df[~msk]

# Display the first few rows of the training set
print("Training set:")
print(train.head())

# Display the first few rows of the test set
print("\nTest set:")
print(test.head())
