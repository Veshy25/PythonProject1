###PythonProject1, Analysing Marketing Campaign Data

#==========================================================================
## 1)Importing and Installing Required Libraries
#==========================================================================
# Import Libraries 
import subprocess
import sys

# List of packages to install  
packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy', 'scikit-learn', 'statsmodels']  
  
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
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm


# Apply seaborn style
sns.set_style(seaborn_style)

# Load the CSV file
file_path = "marketing_campaign.csv"
df = pd.read_csv(file_path, sep=';')

#===========================================================================
## 2) Basic Data Exploration, Visualization & Basic Data Cleaning 
#===========================================================================

# View First 5 Rows of Dataset 
print("🔍 First 5 rows of the dataset:\n")                               
print(df.head())

# Get Dataset Info
print("\n📊 Dataset Info:\n")                                           
print(df.info())

# Check for Missing Values
print("\n❓ Missing Values:\n")                                           
print(df.isnull().sum())

# Remove the rows where there is at least 1 missing value 
df_cleaned = df.dropna()

# Optional: Check how many rows were removed
print(f"🧹 Rows before cleaning: {df.shape[0]}")
print(f"🧼 Rows after cleaning:  {df_cleaned.shape[0]}")

# Descriptive Statistics
print("\n📈 Descriptive Statistics:\n")                                  
print(df_cleaned.describe())

# Plot Correlation Heatmap
plt.figure(figsize=(12,6))                                                
plt.title('Correlation Heatmap of Marketing Campaign Data')
sns.heatmap(df_cleaned.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.show()

# Plot Distribution of Numerical Features
# Get Numerical Columns 
numerical_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns   

# Filter out binary columns and 'ID'
numeric_cols_filtered = [
    col for col in numerical_cols 
    if df_cleaned[col].nunique() > 2 and col.lower() != 'id'
]

# Set up subplot grid
num_cols = len(numeric_cols_filtered)
cols = 3  # Number of columns in subplot grid
rows = math.ceil(num_cols / cols)

# Plot all distributions in one figure
plt.figure(figsize=(cols * 5, rows * 4))

for i, col in enumerate(numeric_cols_filtered, 1):
    plt.subplot(rows, cols, i)
    sns.histplot(df_cleaned[col], kde=True, bins=30)
    plt.title(f"{col}")
    plt.xlabel("")
    plt.ylabel("")

plt.tight_layout()
plt.suptitle("📊 Distribution of Numerical Features (Excluding Binary & ID)", fontsize=16, y=1.02)
plt.show()

# Plot Categorical Columns 
# Identify Categorical Columns (excluding 'Dt_Customer')
categorical_cols = [col for col in df_cleaned.select_dtypes(include='object').columns if col != 'Dt_Customer']

# Set up subplot grid layout
num_cols = len(categorical_cols)
cols = 2  # Number of columns in the grid
rows = math.ceil(num_cols / cols)

plt.figure(figsize=(cols * 5, rows * 4))

# Plot Count of Each Categorical Column
for i, col in enumerate(categorical_cols, 1):
    plt.subplot(rows, cols, i)
    sns.countplot(data=df_cleaned, x=col, order=df_cleaned[col].value_counts().index)
    plt.title(f"{col}")
    plt.xticks(rotation=45)
    plt.xlabel("")
    plt.ylabel("")

plt.tight_layout()
plt.suptitle("📋 Count of Categorical Features (Excl. Dt_Customer)", fontsize=16, y=1.02)
plt.show()

#===========================================================================
## 3) Data Cleaning & Data Transformation
#===========================================================================

# Check how many duplicate rows exist
num_duplicates = df_cleaned.duplicated().sum()
print(f"🔍 Number of duplicate rows in the dataset: {num_duplicates}")  # No Duplicates , No Further updates needed

# 'YOLO', 'Alone', and 'Absurd' are likely non-serious or inconsistent entries,
# so we'll group them under 'Single' for meaningful analysis.
print("🔍 Unique values in 'Marital_Status':")
print(df_cleaned['Marital_Status'].unique())

print("\n📊 Value counts in 'Marital_Status':")      
print(df_cleaned['Marital_Status'].value_counts())   

# Replace unusual values with 'Single'
df_cleaned.loc[:, 'Marital_Status'] = df_cleaned['Marital_Status'].replace({
    'YOLO': 'Single',
    'Absurd': 'Single',
    'Alone': 'Single'
})

# Verify the replacement
print("🔎 Unique values in 'Marital_Status' after replacement:")
print(df_cleaned['Marital_Status'].unique())
print("\n📊 Value counts in 'Marital_Status' after replacement:")
print(df_cleaned['Marital_Status'].value_counts())

# Check for Outliers through Visualization
# Columns to visualize and detect outliers
num_vars = [
    'Income','Year_Birth', 'Kidhome', 'Teenhome', 'Recency', 
    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 
    'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 
    'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 
    'NumWebVisitsMonth'
]

# Visualize outliers using boxplots
plt.figure(figsize=(18, 20))

for i, col in enumerate(num_vars, 1):
    plt.subplot(4, 4, i)
    sns.boxplot(y=df_cleaned[col], color='skyblue')
    plt.title(col)
    plt.xlabel("")  # Optional: clean layout

plt.tight_layout()
plt.suptitle("📦 Boxplots for Outlier Detection", fontsize=18, y=1.02)
plt.show()

# Removing outliers from Income using IQR
Q1 = df_cleaned['Income'].quantile(0.25)
Q3 = df_cleaned['Income'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_cleaned = df_cleaned[(df_cleaned['Income'] >= lower_bound) & (df_cleaned['Income'] <= upper_bound)]

# Optional: Check new Income distribution
plt.figure(figsize=(4, 4))
sns.boxplot(y=df_cleaned['Income'], color='skyblue')
plt.title("Income After Outlier Removal")
plt.tight_layout()
plt.show()

# Removing outliers from Year_Birth using IQR
Q1 = df_cleaned['Year_Birth'].quantile(0.25)
Q3 = df_cleaned['Year_Birth'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_cleaned = df_cleaned[(df_cleaned['Year_Birth'] >= lower_bound) & (df_cleaned['Year_Birth'] <= upper_bound)]

# Optional: Check new Year_Birth distribution
plt.figure(figsize=(4, 4))
sns.boxplot(y=df_cleaned['Year_Birth'], color='skyblue')
plt.title("Year_Birth After Outlier Removal")
plt.tight_layout()
plt.show()

# Print confirmation
print(f"✅ Shape of dataset after removing outliers in 'Income' and 'Year_Birth': {df_cleaned.shape}")


# Check data type of 'Dt_Customer' and a few of its values 
print("\n🔍 Data type of 'Dt_Customer':")
print(df_cleaned['Dt_Customer'].dtype)
print(df_cleaned['Dt_Customer'].head())
# Convert 'Dt_Customer' to datetime format from object string type
df_cleaned.loc[:, 'Dt_Customer'] = pd.to_datetime(df_cleaned['Dt_Customer'], errors='coerce')


# Total Campaigns Accepted Transformation
# Creating a new column to sum the accepted campaigns 
df_cleaned.loc[:, 'Total_Campaigns_Accepted'] = df_cleaned[[
    'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 
    'AcceptedCmp4', 'AcceptedCmp5', 'Response'
]].sum(axis=1)

# Age Group Transformation
## Create Age column
df_cleaned.loc[:, 'Age'] = 2025 - df_cleaned['Year_Birth']
# Create Age Group column using pd.cut
df_cleaned.loc[:, 'Age_Group'] = pd.cut(df_cleaned['Age'],
                                        bins=[0, 30, 45, 60, 100],
                                        labels=['Young', 'Adult', 'Middle-Aged', 'Senior'])



#===========================================================================
## 4) Preliminary Analysis and visualisation
#===========================================================================

# Compare Total Purchases by Channel 
channel_cols = ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
df_cleaned[channel_cols].sum().sort_values(ascending=False).plot(kind='bar')
plt.title("Total Purchases by Marketing Channel")
plt.ylabel("Number of Purchases")
plt.show()

# Caimpaign Acceptance vs Channel usage - Check if people who accepted campaigns tend to use certain channels more. 
# Create a mask for customers who accepted at least one campaign
responders = df_cleaned[df_cleaned['Total_Campaigns_Accepted'] > 0]

# Average purchases per channel by responders
responders[channel_cols].mean().plot(kind='bar', color='green')
plt.title("Average Channel Purchases for Campaign Responders")
plt.ylabel("Average Purchases")
plt.show()

# Cross-tab by Age - Which channel works better for which demographic? 
# Average web purchases by age group
df_cleaned.groupby('Age_Group')['NumWebPurchases'].mean().plot(kind='bar')
plt.title("Avg Web Purchases by Age Group")
plt.ylabel("Average")
plt.show()








# ===========================================================================  
## 7) Logistic Regression: Which Channel Predicts Campaign Response : Question 3 
# ===========================================================================

# Create a binary column : 1 if accepted any campaign, 0 otherwise
df_cleaned['Responded'] = (df_cleaned['Total_Campaigns_Accepted'] > 0).astype(int)

# Select Features (x) and Target (y)
X = df_cleaned[channel_cols]  # already defined as ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
y = df_cleaned['Responded']

# Standardize the features for interpretability
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Add intercept to the features 
X_const = sm.add_constant(X_scaled)

# Fit the logistic regression model
logit_model = sm.Logit(y, X_const)
result = logit_model.fit()

# Print detailed summary of the model
print(result.summary())

# Visualize the influence of each channel on campaign response
# Create a DataFrame for coefficients (excluding intercept)
coef_df = pd.DataFrame({
    'Channel': channel_cols,
    'Coefficient': result.params[1:]  # skip intercept
})

# Plot
plt.figure(figsize=(6, 3))
sns.barplot(data=coef_df, x='Coefficient', y='Channel', palette='coolwarm')
plt.axvline(0, color='black', linestyle='--')
plt.title("Channel Influence on Campaign Acceptance")
plt.tight_layout()
plt.show()
