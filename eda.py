%%writefile eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset safely
df = pd.read_csv(
    'DataSet.csv',
    engine='python',
    on_bad_lines='skip'
)

# Clean column names
df.columns = df.columns.str.strip()

# Dataset info
print("=== Dataset Info ===")
print(df.info())
print()

# First rows
print("=== First 5 rows ===")
print(df.head())
print()

# Missing values
print("=== Missing Values ===")
print(df.isnull().sum())
print()

# Numeric summary
print("=== Numeric Summary ===")
print(df.describe())
print()

# Fake vs Real jobs
if 'fraudulent' in df.columns:
    sns.countplot(x='fraudulent', data=df)
    plt.title("Fake vs Real Job Distribution")
    plt.show()

# Numeric distributions
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], bins=20, kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

# Correlation heatmap
numeric_df = df.select_dtypes(include=['int64', 'float64'])
if numeric_df.shape[1] > 1:
    plt.figure(figsize=(10,8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
    plt.title("Feature Correlation Heatmap")
    plt.show()

# Top job titles
if 'title' in df.columns:
    df['title'].value_counts().head(10).plot(kind='bar')
    plt.title("Top Job Titles")
    plt.show()
