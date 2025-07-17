import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cleaned data
df = pd.read_csv("cleaned_churn_data.csv")

# Clean column names (remove leading/trailing whitespaces)
df.columns = df.columns.str.strip()

# Show all column names (optional, for debugging)
print("Available Columns:", df.columns.tolist())

# Separate churned vs non-churned
churned = df[df['Churn'] == 1]
retained = df[df['Churn'] == 0]

# 1. Compare key features by churn
key_features = ['tenure', 'MonthlyCharges', 'Contract', 'InternetService', 'PaymentMethod']

for feature in key_features:
    if feature in df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(data=df, x=feature, hue='Churn', multiple='stack', palette='Set1')
        plt.title(f"Churn vs. {feature}")
        plt.tight_layout()
        plt.show()
    else:
        print(f"⚠ Feature '{feature}' not found in data!")

# 2. Group-wise churn rate
group_features = ['Contract', 'InternetService', 'PaymentMethod']

for feature in group_features:
    if feature in df.columns:
        churn_rate = df.groupby(feature)['Churn'].mean().sort_values(ascending=False)
        plt.figure(figsize=(6, 4))
        sns.barplot(x=churn_rate.index, y=churn_rate.values)
        plt.title(f'Churn Rate by {feature}')
        plt.ylabel('Churn Rate')
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()
    else:
        print(f"⚠ Column '{feature}' not found for groupby analysis!")

# 3. Print churn rate by contract type
if 'Contract' in df.columns:
    contract_churn = df.groupby('Contract')['Churn'].mean()
    print("--- Churn Rate by Contract Type ---")
    print(contract_churn)
else:
    print("⚠ 'Contract' column not found!")

# 4. Tenure bins and churn
if 'tenure' in df.columns:
    df['tenure_group'] = pd.cut(df['tenure'], bins=[0, 6, 12, 24, 48, 72], labels=['0-6', '7-12', '13-24', '25-48', '49-72'])
    tenure_churn = df.groupby('tenure_group')['Churn'].mean()

    plt.figure(figsize=(6, 4))
    tenure_churn.plot(kind='bar', color='orange')
    plt.title("Churn Rate by Tenure Group")
    plt.ylabel("Churn Rate")
    plt.tight_layout()
    plt.show()
else:
    print("⚠ 'tenure' column not found!")