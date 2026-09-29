import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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

# ── Step 3: Encode categorical columns with get_dummies ──────────────────────
cat_cols = ['gender', 'PhoneService', 'InternetService', 'Contract',
            'PaperlessBilling', 'PaymentMethod']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ── Step 4: Separate features and target ─────────────────────────────────────
X = df.drop(columns=['Churn'])
y = df['Churn']

# ── Step 5: Train/test split — 80/20, random_state=42 ────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Step 6: Train Logistic Regression ────────────────────────────────────────
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ── Step 7: Print accuracy score ─────────────────────────────────────────────
y_pred = model.predict(X_test)
print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))
print()

# ── Step 8: Print classification report ──────────────────────────────────────
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Stay', 'Churn']))
