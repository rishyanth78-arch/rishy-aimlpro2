import pandas as pd
from sklearn.linear_model import LogisticRegression

# ── Step 1: Load the dataset ──────────────────────────────────────────────────
df = pd.read_csv('churnguard_data.csv')

# ── Apply all Task 2 cleaning steps ──────────────────────────────────────────

# Drop customerID
df = df.drop(columns=['customerID'])

# Remove duplicates
df = df.drop_duplicates()

# Strip whitespace
df['gender'] = df['gender'].str.strip()
df['PaymentMethod'] = df['PaymentMethod'].str.strip()

# Standardise casing
df['Churn'] = df['Churn'].str.strip().str.title()
df['PhoneService'] = df['PhoneService'].str.strip().str.title()
df['PaperlessBilling'] = df['PaperlessBilling'].str.strip().str.title()

# Fix Contract
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

# Fix InternetService
internet_map = {
    'dsl':          'DSL',
    'fiber optic':  'Fiber optic',
    'fibre optic':  'Fiber optic',
    'fiberoptic':   'Fiber optic',
    'no':           'No',
    'none':         'No',
}
df['InternetService'] = df['InternetService'].str.strip().str.lower().map(internet_map)

# Fix TotalCharges
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Remove invalid tenure
df = df[df['tenure'] > 0]

# Remove outlier MonthlyCharges
df = df[(df['MonthlyCharges'] >= 10) & (df['MonthlyCharges'] <= 200)]

# Fill missing values
df['MonthlyCharges'] = df['MonthlyCharges'].fillna(df['MonthlyCharges'].mean())
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].mean())
df['tenure'] = df['tenure'].fillna(round(df['tenure'].median()))

# ── Step 2: Encode target column ─────────────────────────────────────────────
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# ── Step 3: Select only the five numeric/ordinal features for this task ───────
features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Contract']

# Encode Contract as ordinal: Month-to-month=0, One year=1, Two year=2
contract_ordinal = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
df['Contract'] = df['Contract'].map(contract_ordinal)

X = df[features]
y = df['Churn']

# ── Step 4: Retrain Logistic Regression on the full cleaned dataset ───────────
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ── Step 5: Collect user inputs ───────────────────────────────────────────────
print("Enter customer details to predict churn:\n")

tenure         = int(input("Enter tenure (months): "))
monthly_charges = float(input("Enter Monthly Charges: "))
total_charges  = float(input("Enter Total Charges: "))
senior_citizen = int(input("Senior Citizen? (1 = Yes, 0 = No): "))
contract       = int(input("Contract type (0 = Month-to-month, 1 = One year, 2 = Two year): "))

# ── Step 6: Build input DataFrame and predict ─────────────────────────────────
input_data = pd.DataFrame([[tenure, monthly_charges, total_charges,
                             senior_citizen, contract]],
                          columns=features)

prediction = model.predict(input_data)[0]

# ── Step 7: Print result ──────────────────────────────────────────────────────
if prediction == 1:
    print("\nPrediction: This customer is likely to CHURN.")
else:
    print("\nPrediction: This customer is likely to STAY.")
