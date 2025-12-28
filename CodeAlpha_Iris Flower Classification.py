# Iris Flower Classification using Kaggle Dataset
# CodeAlpha Data Science Internship - Task 1

# Import libraries
import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Download dataset from Kaggle
path = kagglehub.dataset_download("saurabh00007/iriscsv")
print("Dataset downloaded at:", path)

# Load dataset
csv_file = os.path.join(path, "Iris.csv")
df = pd.read_csv(csv_file)

# Display dataset
print("\nFirst 5 rows of dataset:")
print(df.head())

# Drop Id column (not useful for prediction)
df.drop(columns=["Id"], inplace=True)

# Encode target labels
le = LabelEncoder()
df["Species"] = le.fit_transform(df["Species"])

# Dataset info
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Data Visualization
sns.pairplot(df, hue="Species")
plt.show()

# Split features and target
X = df.drop("Species", axis=1)
y = df["Species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model training
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
