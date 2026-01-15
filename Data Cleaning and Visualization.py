# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("data.csv")
print("First 5 records:")
print(data.head())

# Display dataset information
print("\nDataset Information:")
print(data.info())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Handle missing values using mean (numeric columns only)
data.fillna(data.mean(numeric_only=True), inplace=True)

# Descriptive statistics
print("\nDescriptive Statistics:")
print(data.describe())

# Outlier detection using IQR
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

data_cleaned = data[~((data < (Q1 - 1.5 * IQR)) | 
                      (data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Normalization (Z-score normalization)
data_normalized = (data_cleaned - data_cleaned.mean()) / data_cleaned.std()

# Histogram
data_normalized.hist(figsize=(10, 8))
plt.suptitle("Histogram of Normalized Data")
plt.show()

# Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(data=data_normalized)
plt.title("Boxplot of Normalized Data")
plt.show()

# Scatter plot (first two columns)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data_normalized.iloc[:, 0],
                y=data_normalized.iloc[:, 1])
plt.title("Scatter Plot")
plt.show()
