import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Dataset shape
print("Shape of dataset:", df.shape)

# First 5 rows
print(df.head())

# Data info
print("\n--- Dataset Info ---")
print(df.info())

# Churn value counts
print("\n--- Churn Distribution ---")
print(df['Churn'].value_counts(normalize=True))

# step 1 ends here

# --------------------------
# Step 2: Data Cleaning
# --------------------------

# Convert 'TotalCharges' to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Check for missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Drop rows with missing TotalCharges
df = df.dropna()
print("Shape after dropping nulls:", df.shape)

# Drop 'customerID' column
df = df.drop('customerID', axis=1)

# Convert 'Churn' from Yes/No to 1/0
df['Churn'] = df['Churn'].apply(lambda x: 1 if x.strip() == 'Yes' else 0)

# Final data preview
print("\n--- Cleaned Data Preview ---")
print(df.head())

# Updated churn distribution
print("\n--- Updated Churn Distribution ---")
print(df['Churn'].value_counts(normalize=True))

# step 2 ends here

# --------------------------
# Step 3: Exploratory Data Analysis (EDA)
# --------------------------

# Set seaborn style
sns.set(style='whitegrid')

# Churn distribution bar plot
plt.figure(figsize=(6,4))
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Churn by gender
plt.figure(figsize=(6,4))
sns.countplot(x='gender', hue='Churn', data=df)
plt.title("Churn by Gender")
plt.show()

# Churn by contract type
plt.figure(figsize=(6,4))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Churn by Contract Type")
plt.show()

# Box plot for Monthly Charges
plt.figure(figsize=(6,4))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

# Correlation Heatmap for numerical columns
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

