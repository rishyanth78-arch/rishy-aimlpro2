import pandas as pd

# Load the dataset
df = pd.read_csv('churnguard_data.csv')

# 1. Print the shape of the dataset
print("Dataset Shape:")
print(df.shape)
print()

# 2. Print the first 5 rows
print("First 5 Rows:")
print(df.head())
print()

# 3. Print column names and data types using .info()
print("Column Names and Data Types:")
df.info()
print()

# 4. Print missing value count for each column
print("Missing Value Count per Column:")
print(df.isnull().sum())
print()

# 5. Print number of duplicate rows
print("Number of Duplicate Rows:")
print(df.duplicated().sum())
print()

# 6. Print value counts of the Churn column
print("Value Counts for Churn Column:")
print(df['Churn'].value_counts())
print()

# 7. Print unique values in the Contract column
print("Unique Values in Contract Column:")
print(df['Contract'].unique())
