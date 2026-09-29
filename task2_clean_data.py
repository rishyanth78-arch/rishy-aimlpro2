import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv('churnguard_data.csv')

# Step 2: Drop the customerID column
df = df.drop(columns=['customerID'])

# Step 3: Remove duplicate rows
df = df.drop_duplicates()

# Step 4: Strip whitespace from gender and PaymentMethod
df['gender'] = df['gender'].str.strip()
df['PaymentMethod'] = df['PaymentMethod'].str.strip()

# Step 5: Standardise casing — convert Churn, PhoneService, and PaperlessBilling to title case
df['Churn'] = df['Churn'].str.strip().str.title()
df['PhoneService'] = df['PhoneService'].str.strip().str.title()
df['PaperlessBilling'] = df['PaperlessBilling'].str.strip().str.title()

# Step 6: Fix Contract — map all variations to one of three valid values
contract_map = {
    'month-to-month': 'Month-to-month',
    'monthly':        'Month-to-month',
    'month to month': 'Month-to-month',
    'one year':       'One year',
    '1 year':         'One year',
    'oneyear':        'One year',
    'two year':       'Two year',
    '2 year':         'Two year',
    'twoyear':        'Two year',
}
df['Contract'] = df['Contract'].str.strip().str.lower().map(contract_map)

# Step 7: Fix InternetService — map all variations to one of three valid values
internet_map = {
    'dsl':          'DSL',
    'fiber optic':  'Fiber optic',
    'fibre optic':  'Fiber optic',
    'fiberoptic':   'Fiber optic',
    'no':           'No',
    'none':         'No',
}
df['InternetService'] = df['InternetService'].str.strip().str.lower().map(internet_map)

# Step 8: Fix TotalCharges — convert to numeric; junk values become NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Step 9: Remove rows where tenure is zero or negative
df = df[df['tenure'] > 0]

# Step 10: Remove rows where MonthlyCharges is less than 10 or greater than 200
df = df[(df['MonthlyCharges'] >= 10) & (df['MonthlyCharges'] <= 200)]

# Step 11: Fill missing values
df['MonthlyCharges'] = df['MonthlyCharges'].fillna(df['MonthlyCharges'].mean())
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
df['tenure'] = df['tenure'].fillna(round(df['tenure'].median()))

# Step 12: Print the shape of the cleaned DataFrame
print("Cleaned Dataset Shape:")
print(df.shape)
print()

# Step 13: Print missing value counts to confirm all issues are resolved
print("Missing Value Counts After Cleaning:")
print(df.isnull().sum())
