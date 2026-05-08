import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("📥 Loading dataset...")

# -------------------------------
# 1. Load Dataset
# -------------------------------
DATA_PATH = "model/WA_Fn-UseC_-Telco-Customer-Churn.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("❌ Dataset not found. Check path!")

df = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded")
print("Shape:", df.shape)

# -------------------------------
# 2. Data Cleaning
# -------------------------------

# Drop customerID (not useful)
df.drop("customerID", axis=1, inplace=True)

# Fix TotalCharges (convert to numeric)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Fill missing values
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# Convert target column
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Fix SeniorCitizen (0/1 already but ensure int)
df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

print("🧹 Data cleaning done")

# -------------------------------
# 3. Encode Categorical Features
# -------------------------------
label_encoders = {}

for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("🔤 Encoding done")

# -------------------------------
# 4. Split Features & Target
# -------------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("📊 Train/Test split done")

# -------------------------------
# 5. Scaling
# -------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("⚖️ Scaling done")

# -------------------------------
# 6. Train Model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("🤖 Model trained")

# -------------------------------
# 7. Evaluate Model
# -------------------------------
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\n📈 MODEL PERFORMANCE")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------
# 8. Feature Importance
# -------------------------------
feature_importance = dict(
    zip(feature_names, model.feature_importances_)
)

# Sort by importance
feature_importance = dict(
    sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
)

print("\n🔥 Top 5 Important Features:")
for k, v in list(feature_importance.items())[:5]:
    print(f"{k}: {round(v, 4)}")

# -------------------------------
# 9. Save Everything
# -------------------------------
print("\n💾 Saving model files...")

pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(scaler, open("model/scaler.pkl", "wb"))
pickle.dump(label_encoders, open("model/encoders.pkl", "wb"))
pickle.dump(feature_importance, open("model/feature_importances.pkl", "wb"))
pickle.dump(feature_names, open("model/feature_names.pkl", "wb"))

print("✅ All files saved successfully!")

# -------------------------------
# DONE
# -------------------------------
print("\n🎉 Training Complete with REAL DATA!")