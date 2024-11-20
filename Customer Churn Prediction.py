#!/usr/bin/env python
# coding: utf-8

# ### The goal of this study is to analyze the dataset and develop a predictive model for customer churn, helping the company proactively identify at-risk customers and implement retention strategies.

# ###### Dataset Overview
# The dataset includes the following attributes for each customer:
# CustomerID: Unique identifier for each customer.
# Age: Age of the customer.
# Gender: Gender of the customer.
# Tenure: Duration of the customer relationship (e.g., months or years).
# Usage Frequency: Frequency of service usage (e.g., monthly or weekly).
# Support Calls: Number of support calls made by the customer.
# Payment Delay: Indicates whether the customer had payment delays (yes/no or number of days).
# Subscription Type: Type of subscription plan (e.g., Basic, Standard, Premium).
# Contract Length: Duration of the subscription contract (e.g., 1 month, 12 months).
# Total Spend: Total monetary value spent by the customer.
# Last Interaction: The time since the last customer interaction.
# Churn: Target variable indicating if the customer has churned (yes/no).

# ###### Problem Statement
# 
# Customer churn impacts the revenue and growth of the company. Understanding the factors contributing to churn enables the development of data-driven strategies to enhance customer retention.

# ###### Methodology
# 1.Data Preprocessing
# 2.Exploratory Data Analysis (EDA)
# 3.Feature Engineering
# 4.Model Selection
# 5.Model Evaluation
# 6.Interpretability
# 7.Implementation and Monitoring

# ###### 1. Import Libraries
# pandas - Data loading, manipulation, and basic exploration.
# numpy - Numerical operations and efficient computation.
# seaborn - Advanced, intuitive data visualization.
# matplotlib.pyplot - Core library for creating static, interactive, and animated visualizations.
# scikit-learn - Comprehensive machine learning and evaluation framework.
# joblib - For saving and reloading machine learning models.

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve


# ###### 2. Load the Dataset

# In[2]:


# Load dataset
file_path = "/Users/sheeba/Desktop/ML- Case Study/customer_churn_dataset-training-master.csv"
# Replace with your dataset path
df = pd.read_csv(file_path)

# Display first few rows
df.head()


# ###### 3. Data Preprocessing
# Handle Missing Values
# ###### Handle Missing Values
# Missing values can degrade the performance of machine learning models by affecting feature relationships and data distributions.
# ###### Drop Missing Values/Fill Missing Values (Imputation)
# Forward fill assumes that the last known value remains valid until a new value is observed.
# ##### Encode Categorical Variables
# 
# ##### Scale Numerical Features

# In[3]:


# Check for missing values
print(df.isnull().sum())

# Fill or drop missing values
df.fillna(method='ffill', inplace=True)  # Example: forward fill


# ##### Encode Categorical Variables
# Data encoding is essential in machine learning because many algorithms require numerical input. If the data contains categorical or textual features, encoding converts these into a numerical format that algorithms can process effectively.
# Most ML algorithms cannot handle non-numeric data directly, as they rely on mathematical operations like distance or probability, which are undefined for categorical values.
# Encoding allows categorical features to be represented numerically while preserving the relationships or distinctions between categories.

# In[21]:


# Encode categorical variables
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Payment Delay'] = label_encoder.fit_transform(df['Payment Delay'])
df['Subscription Type'] = label_encoder.fit_transform(df['Subscription Type'])
df['Churn'] = label_encoder.fit_transform(df['Churn'])  # Target variable
df['Contract Length'] = label_encoder.fit_transform(df['Contract Length'])


# #####  Features to scale
# Scaling data is an essential preprocessing step in many machine learning workflows. It ensures that all numerical features have the same scale, preventing features with larger ranges from dominating the model and improving its performance.

# In[22]:


# Features to scale
features_to_scale = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 'Total Spend']

scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])


# ### 4. Exploratory Data Analysis (EDA)
# Visualize Correlation Heatmap
# A correlation heatmap is a visual representation of the relationships between numerical variables in a dataset. It helps identify patterns, dependencies, and redundancies, which are crucial for feature selection, engineering, and improving model performance.
# 
# Identifying Strong Relationships
# 
# High Positive Correlation:
# Features with a high correlation (closer to +1) move in the same direction.
# Example: Tenure and Total Spend may have a positive correlation in a churn dataset, as longer tenure customers might spend more.
# High Negative Correlation:
# Features with a strong negative correlation (closer to -1) move in opposite directions.
# Example: Usage Frequency and Churn might be negatively correlated, as frequent usage reduces the likelihood of churn.
# 
# ###### Relationship Between Churn and Input Features
# This helps identify which features have a strong or weak relationship with the target (Churn).
# Numerical Features: Use correlation (e.g., Pearson, Spearman).
# Example: Tenure vs. Churn (check correlation coefficient).
# Categorical Features: Use statistical tests or visualizations.
# Example: Use a bar plot or chi-square test for Gender vs. Churn.
# ###### Relationships Between Input Features (Inter-Feature Relationships)
# Helps identify multicollinearity or redundant features.
# Use a correlation heatmap or Variance Inflation Factor (VIF).
# Example: If Total Spend and Tenure are highly correlated, one might be redundant.
# Decide whether to drop one, combine them, or apply dimensionality reduction techniques (e.g., PCA).
# ###### Features with strong positive or negative correlations (or associations).
# Example:
# Usage Frequency: A high negative correlation suggests frequent usage reduces churn.
# Support Calls: A high positive correlation may indicate dissatisfaction leading to churn.
# 

# In[23]:


# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()


# #### Analyze Churn Distribution
# 
# ###### Balanced Distribution:
# If the churn distribution is relatively balanced (e.g., 50% churned, 50% stayed), the model training can proceed without special handling for class imbalance.
# ###### Imbalanced Distribution:
# Significant class imbalance can make models biased toward the majority class (e.g., predicting "No churn" for most cases).
# ###### Strategies to handle imbalance:
# Resampling: Oversample the minority class (churned) or undersample the majority class (stayed).
# Synthetic Data Generation: Use methods like SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic samples for the minority class.
# Adjusting Class Weights: Some models (e.g., logistic regression, random forests) allow you to set higher weights for the minority class.

# In[24]:


# Churn distribution
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()


# ##### Compare Features by Churn Status
#  Comparing features by churn status (i.e., analyzing how different features behave for churned vs. non-churned customers) provides critical insights into customer behavior and helps build more accurate prediction models.

# In[25]:


# Boxplot for Total Spend by Churn
sns.boxplot(x='Churn', y='Total Spend', data=df)
plt.title("Total Spend by Churn")
plt.show()


# ### 5. Feature Engineering
# Create New Features
# 
# Creating new features is a critical part of feature engineering and can significantly enhance the performance of machine learning models. By generating additional features, you provide the model with more relevant information, which can help improve accuracy, reduce overfitting, and make the model more interpretable.

# In[26]:


# Example: Average Spend per Month
#This feature calculates how much a customer spends on average per month by dividing the Total Spend by the Tenure (the number of months the customer has been with the service).
#The small value (1e-5) is added to avoid division by zero errors, in case a customer has a Tenure of 0

df['Avg Spend per Month'] = df['Total Spend'] / (df['Tenure'] + 1e-5)  # Avoid division by zero

# Example: Interaction Recency (assuming Last Interaction is in days)
df['Interaction Recency'] = df['Last Interaction']  # Adjust based on dataset


# ### 6. Train-Test Split

# In[27]:


# Split features and target
X = df.drop(['CustomerID', 'Churn'], axis=1)  # Drop unnecessary columns
y = df['Churn']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# ### 7. Train a Predictive Model

# In[28]:


# Train Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# ### 8. Evaluate the Model

# In[29]:


# Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# In[30]:


# Classification Report
print(classification_report(y_test, y_pred))


# In[31]:


# ROC Curve
y_pred_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_pred_prob):.2f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


# ### 9. Feature Importance

# In[32]:


# Feature Importance
importances = model.feature_importances_
feature_names = X.columns

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance")
plt.show()


# ### 10. Save and Deploy Model

# In[33]:


import joblib

# Save the model
joblib.dump(model, "churn_model.pkl")

# Load the model (if needed)
loaded_model = joblib.load("churn_model.pkl")


# In[ ]:




