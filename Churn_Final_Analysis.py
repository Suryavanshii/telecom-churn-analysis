# Project: Customer Churn Analysis & Retention Strategy
# Author: Rajkumar Suryavanshi
# Description: Final analysis script with churn rate insights and business strategy
# Date: 2025-07-17

import pandas as pd

# Load data
df = pd.read_csv("cleaned_churn_data.csv")
df.columns = df.columns.str.strip()

# Overall churn rate
churn_rate = df['Churn'].mean()
print(f"\n✅ Overall Churn Rate: {churn_rate:.2%}")

# Contract Type Analysis (based on one-hot encoding)
print("\n📊 Churn Rate by Contract Type:")
for col in ['Contract_Month-to-month', 'Contract_One year', 'Contract_Two year']:
    rate = df[df[col] == 1]['Churn'].mean()
    print(f"{col.replace('Contract_', '')}: {rate:.2%}")

# Payment Method Analysis
print("\n💳 Churn Rate by Payment Method:")
for col in ['PaymentMethod_Electronic check', 'PaymentMethod_Mailed check',
            'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)']:
    rate = df[df[col] == 1]['Churn'].mean()
    print(f"{col.replace('PaymentMethod_', '')}: {rate:.2%}")

# Internet Service Analysis
print("\n🌐 Churn Rate by Internet Service:")
for col in ['InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No']:
    rate = df[df[col] == 1]['Churn'].mean()
    print(f"{col.replace('InternetService_', '')}: {rate:.2%}")

# Tenure Grouping (fixing NaNs)
df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 48, 72], 
                            labels=['0-6', '7-12', '13-24', '25-48', '49-72'])

tenure_churn = df.groupby('tenure_group')['Churn'].mean()
print("\n📈 Churn Rate by Tenure Group:")
print(tenure_churn)

# 🔚 Recommendations
print("\n--- 🔍 Strategic Recommendations ---")
print("1. Offer loyalty benefits to month-to-month users (highest churn).")
print("2. Improve experience for Fiber optic users.")
print("3. Focus retention efforts on early tenure groups (0-6 months).")
print("4. Encourage secure auto-pay methods to reduce churn from 'Electronic check' users.")
print("5. Upsell bundled services to add value for long-term retention.")