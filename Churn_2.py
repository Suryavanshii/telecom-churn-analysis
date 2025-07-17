# Churn_2.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Drop customerID column (not useful for prediction)
df.drop("customerID", axis=1, inplace=True)

# Convert TotalCharges to numeric (some entries are empty strings)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# Check missing values
print("--- Missing Values Before ---")
print(df.isnull().sum())

# Drop rows with missing TotalCharges
df.dropna(inplace=True)

# Encode target variable: Churn (Yes → 1, No → 0)
df["Churn"] = df["Churn"].map({'Yes': 1, 'No': 0})

# Identify categorical columns
cat_cols = df.select_dtypes(include='object').columns.tolist()

# Label Encode binary categorical columns, One-Hot Encode rest
le = LabelEncoder()
for col in cat_cols:
    if df[col].nunique() == 2:
        df[col] = le.fit_transform(df[col])
    else:
        df = pd.get_dummies(df, columns=[col])

# Scale numerical features
scaler = StandardScaler()
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
df[num_cols] = scaler.fit_transform(df[num_cols])

# Show final shape and a preview
print("\n--- Data After Preprocessing ---")
print(df.shape)
print(df.head())

# Optional: Save cleaned dataset for modeling
df.to_csv("cleaned_churn_data.csv", index=False)
print("\n✅ Preprocessing complete. Cleaned data saved as 'cleaned_churn_data.csv'")