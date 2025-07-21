###PythonProject1, Analysing Marketing Campaign Data

## 1)Importing and Installing Required Libraries
# Import Libraries 
import subprocess
import sys

# List of packages to install
packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy', 'scikit-learn']

# Loop through and install each
for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Set Visualization Style 
seaborn_style = 'whitegrid'

# Import necessary libraries after installation
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import math

# Apply seaborn style
sns.set_style(seaborn_style)

# Load the CSV file
file_path = "marketing_campaign.csv"
df = pd.read_csv(file_path, sep=';')

## 2) Basic Data Exploration and Visualization
print("🔍 First 5 rows of the dataset:\n")   #View First 5 Rows of Dataset 
print(df.head())

print("\n📊 Dataset Info:\n")                # 
print(df.info())

print("\n❓ Missing Values:\n")              # Check for Missing Values 
print(df.isnull().sum())

print("\n📈 Descriptive Statistics:\n")      # Descriptive Statistics
print(df.describe())

plt.figure(figsize=(12,6))                   # Plot Correlation Heatmap
plt.title('Correlation Heatmap of Marketing Campaign Data')
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

# Get numeric columns                        # Plot Distribution of Numerical Features
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Filter out binary columns and 'ID'
numeric_cols_filtered = [
    col for col in numerical_cols 
    if df[col].nunique() > 2 and col.lower() != 'id'
]

# Set up subplot grid
num_cols = len(numeric_cols_filtered)
cols = 3  # Number of columns in subplot grid
rows = math.ceil(num_cols / cols)

# Plot all distributions in one figure
plt.figure(figsize=(cols * 5, rows * 4))

for i, col in enumerate(numeric_cols_filtered, 1):
    plt.subplot(rows, cols, i)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"{col}")
    plt.xlabel("")
    plt.ylabel("")

plt.tight_layout()
plt.suptitle("📊 Distribution of Numerical Features (Excluding Binary & ID)", fontsize=16, y=1.02)
plt.show()
