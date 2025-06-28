import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Style for better aesthetics
plt.style.use('seaborn-v0_8-darkgrid')

# Load and prepare the Iris Dataset
iris = load_iris(as_frame=True)

# X contains the features
X = iris.data

# y contains the target
y = iris.target

# Get the names of the target classes and features
target_names = iris.target_names
feature_names = iris.feature_names

print("Iris Dataset Overview")
print(f"\nFeature names: {feature_names}")
print(f"Target class names: {target_names}")
print(f"Dataset shape: {X.shape} (samples, features)")

# Splitting the dataset (training & testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, stratify=y)

print("\n Dataset Split Information")
print(f"Shape of X_train (training features): {X_train.shape}")
print(f"Shape of X_test (testing features): {X_test.shape}")
print(f"Shape of y_train (training targets): {y_train.shape}")
print(f"Shape of y_test (testing targets): {y_test.shape}")

# Building and training KNN classifier
k_value = 5
knn_classifier = KNeighborsClassifier(n_neighbors=k_value, metric = 'euclidean')

print(f"\n Initializing and Training KNN Classifier (K={k_value}, Metric = Euclidean)")

knn_classifier.fit(X_train, y_train)
print("KNN Classifier training complete.")

# Making predictions
y_pred = knn_classifier.predict(X_test)

print("\n Predictions on Test Data")
print(f"First 10 true labels (y_test): {y_test.values[:10]}")
print(f"First 10 predicted labels (y_pred): {y_pred[:10]}")

# Evaluating KNN model
print("\n Evaluating KNN Model Performance")

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n Confusion Matrix")
print(cm)

# Visualize Confusion Matrix with Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title(f'Confusion Matrix for KNN (K={k_value})', fontsize=14)
plt.show()

# Classification report
report = classification_report(y_test, y_pred, target_names=target_names)
print("\n Classification Report")
print(report)

print("\n KNN Model Evaluation Complete!")