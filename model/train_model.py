import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Generate synthetic Indian telecom churn dataset
np.random.seed(42)
n = 2000

genders = np.random.choice(['Male', 'Female'], n)
ages = np.random.randint(18, 70, n)
tenures = np.random.randint(1, 72, n)
monthly_charges = np.random.uniform(299, 2999, n)
total_charges = monthly_charges * tenures + np.random.uniform(-100, 100, n)
contract_types = np.random.choice(['Monthly', 'Yearly'], n, p=[0.65, 0.35])
internet_services = np.random.choice(['Fiber', 'DSL', 'None'], n, p=[0.45, 0.40, 0.15])
payment_methods = np.random.choice(['UPI', 'Credit Card', 'Debit Card', 'Cash'], n)
support_calls = np.random.randint(0, 10, n)
senior_citizens = np.random.choice([0, 1], n, p=[0.85, 0.15])

# Churn logic
churn_prob = (
    (tenures < 12) * 0.3 +
    (monthly_charges > 1500) * 0.2 +
    (contract_types == 'Monthly') * 0.15 +
    (support_calls > 5) * 0.2 +
    (internet_services == 'Fiber') * 0.1 +
    (senior_citizens == 1) * 0.05
)
churn_prob = np.clip(churn_prob, 0, 1)
churn = (np.random.rand(n) < churn_prob).astype(int)

df = pd.DataFrame({
    'gender': genders,
    'age': ages,
    'tenure': tenures,
    'monthly_charges': monthly_charges,
    'total_charges': total_charges,
    'contract_type': contract_types,
    'internet_service': internet_services,
    'payment_method': payment_methods,
    'support_calls': support_calls,
    'senior_citizen': senior_citizens,
    'churn': churn
})

# Save dataset
df.to_csv(os.path.join(os.path.dirname(__file__), 'churn_dataset.csv'), index=False)

# Encode categoricals
le_gender = LabelEncoder()
le_contract = LabelEncoder()
le_internet = LabelEncoder()
le_payment = LabelEncoder()

df['gender_enc'] = le_gender.fit_transform(df['gender'])
df['contract_enc'] = le_contract.fit_transform(df['contract_type'])
df['internet_enc'] = le_internet.fit_transform(df['internet_service'])
df['payment_enc'] = le_payment.fit_transform(df['payment_method'])

feature_cols = ['gender_enc', 'age', 'tenure', 'monthly_charges', 'total_charges',
                'contract_enc', 'internet_enc', 'payment_enc', 'support_calls', 'senior_citizen']

X = df[feature_cols]
y = df['churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

feature_names = ['Gender', 'Age', 'Tenure', 'Monthly Charges', 'Total Charges',
                 'Contract Type', 'Internet Service', 'Payment Method', 'Support Calls', 'Senior Citizen']
importances = dict(zip(feature_names, model.feature_importances_.tolist()))

model_dir = os.path.dirname(__file__)
with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
    pickle.dump(model, f)
with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(model_dir, 'encoders.pkl'), 'wb') as f:
    pickle.dump({'gender': le_gender, 'contract': le_contract,
                 'internet': le_internet, 'payment': le_payment}, f)
with open(os.path.join(model_dir, 'feature_importances.pkl'), 'wb') as f:
    pickle.dump(importances, f)

print("Model, scaler, encoders, and feature importances saved.")
