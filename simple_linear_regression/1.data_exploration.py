import pandas as pd
import matplotlib.pyplot as plt
import requests

# Replace 'path' with the URL of the CSV file
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

# Downloading the file using requests library from the given URL
response = requests.get(url)  # Making a GET request to the specified URL
filename = "FuelConsumption.csv"  # Name of the file to save the downloaded data

# Writing the contents of the downloaded file to a local file
with open(filename, "wb") as f:  # Opening a file in binary write mode
    f.write(response.content)  # Writing the content of the HTTP response to the file

# Reading the CSV file into a DataFrame using pandas
df = pd.read_csv(filename)  # Creating a DataFrame by reading the CSV file

# Selecting specific columns for analysis and creating a new DataFrame
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# Save 'cdf' DataFrame to a file for use in the second script
cdf.to_csv('processed_data.csv', index=False)

# Displaying the first few rows of the dataset
print(cdf.head())  # Printing the first few rows of the DataFrame

# Summarizing the statistical data of the DataFrame
print(cdf.describe())  # Printing summary statistics of the numerical columns in the DataFrame

# Generating histograms for selected columns in the DataFrame
viz = cdf[['CYLINDERS', 'ENGINESIZE', 'CO2EMISSIONS', 'FUELCONSUMPTION_COMB']]
viz.hist()  # Plotting histograms for the selected columns
plt.show()  # Displaying the histograms

# Generating scatter plots
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("Emission")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
