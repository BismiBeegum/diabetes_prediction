# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Set style for seaborn plots
sns.set()
import warnings
warnings.filterwarnings('ignore')  # Ignore warnings for cleaner output

# Load the dataset
diabetes_df = pd.read_csv(".../data/diabetes.csv")

# Check for missing values
print(diabetes_df.isnull().sum())

# Create a deep copy to preserve the original dataset
diabetes_df_copy = diabetes_df.copy(deep=True)

# Replace zero values with NaN for certain features where zero is not physiologically valid
cols_to_replace = ['Glucose_level', 'Diastolic_BloodPressure', 'SkinThickness', 'Insulin_level', 'BMI']
diabetes_df_copy[cols_to_replace] = diabetes_df_copy[cols_to_replace].replace(0, np.NaN)

# Plot histograms of all features to understand distribution
diabetes_df_copy.hist(figsize=(20, 20))
plt.show()

# Visualize distribution and outliers in 'Insulin_level'
plt.figure(figsize=(16, 5))
plt.subplot(1, 2, 1)
sns.histplot(diabetes_df_copy['Insulin_level'], kde=True)
plt.title('Distribution of Insulin Level')

plt.subplot(1, 2, 2)
sns.boxplot(x=diabetes_df_copy['Insulin_level'])
plt.title('Boxplot of Insulin Level')
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(diabetes_df_copy.corr(), annot=True, cmap='RdYlGn')
plt.title("Feature Correlation Heatmap")
plt.show()

# Fill missing values with appropriate statistics
diabetes_df_copy['Glucose_level'].fillna(diabetes_df_copy['Glucose_level'].mean(), inplace=True)
diabetes_df_copy['Diastolic_BloodPressure'].fillna(diabetes_df_copy['Diastolic_BloodPressure'].mean(), inplace=True)
diabetes_df_copy['SkinThickness'].fillna(diabetes_df_copy['SkinThickness'].median(), inplace=True)
diabetes_df_copy['Insulin_level'].fillna(diabetes_df_copy['Insulin_level'].median(), inplace=True)
diabetes_df_copy['BMI'].fillna(diabetes_df_copy['BMI'].median(), inplace=True)

# Define feature variables (X) and target variable (y)
X = diabetes_df_copy.drop('Outcome', axis=1)
y = diabetes_df_copy['Outcome']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=12)

# Train a Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=200, random_state=42)
rfc.fit(x_train, y_train)

# Evaluate on training data
train_predictions = rfc.predict(x_train)
print("Training Accuracy Score =", format(accuracy_score(y_train, train_predictions)))

# Evaluate on test data
test_predictions = rfc.predict(x_test)
print("Test Accuracy Score =", format(accuracy_score(y_test, test_predictions)))

# Optional: Print classification report and confusion matrix
# print(classification_report(y_test, test_predictions))
# print(confusion_matrix(y_test, test_predictions))
import pickle
with open('model.sav', 'wb') as model_file:
    pickle.dump(rfc, model_file)
print("Model saved as model.sav")
