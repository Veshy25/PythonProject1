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

# Apply seaborn style
sns.set_style(seaborn_style)

# Load the CSV file
file_path = "marketing_campaign.csv"
df = pd.read_csv(file_path, sep=';')

## 2) Basic Data Exploration and Visualization
print("🔍 First 5 rows of the dataset:\n")
print(df.head())

print("\n📊 Dataset Info:\n")
print(df.info())

print("\n❓ Missing Values:\n")
print(df.isnull().sum())


