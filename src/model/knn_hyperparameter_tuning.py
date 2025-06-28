import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set a style for aesthetics
plt.style.use('seaborn-v0_8-darkgrid')

# Load and prepare the Iris dataset
iris = load_iris(as_frame=True)
X = iris.data
y = iris.target
target_names = iris.target_names
feature_names = iris.feature_names

print(f"Dataset shape: {X.shape} (samples, features)")
print(f"Target class names: {target_names}")

# Splitting the dataset (training & testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, stratify=y)

print("\nInitial Dataset Split Information\n")
print(f"Shape of X_train (training features for GridSearchCV): {X_train.shape}")
print(f"Shape of X_test (truly unseen test features): {X_test.shape}")

# Hyperparameter tuning with GridSearchCV
print("\nStarting GridSearchCV for Hyperparameter Tuning\n")

param_grid = {"n_neighbors": np.arange(1, 21),
              "metric": ["euclidean", "manhattan"]}

# Initialize the base KNeighborsClassifier
knn_base = KNeighborsClassifier()

# Setup GridSearchCV
grid_search = GridSearchCV(estimator = knn_base,
                           param_grid=param_grid,
                           cv=5,
                           scoring="accuracy",
                           verbose=1,
                           n_jobs=-1)

# Fit GridSearchCV to the training data
grid_search.fit(X_train, y_train)
print("\nGridSearchCV Results\n")
print(f"Best cross-validation accuracy achieved: {grid_search.best_score_:.4f}")
print(f"Best parameters found: {grid_search.best_params_}")

# Evaluate the best model with the optimal K and metric
best_knn_model = grid_search.best_estimator_
print(f"\nOptimal KNN Model: {best_knn_model}")

# Make predictions on truly unseen test set
y_pred_optimal = best_knn_model.predict(X_test)

# Calculate and print final evaluation metrics
print("\nEvaluating Optimal KNN Model on Truly Unseen Test Set\n")
accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
cm_optimal = confusion_matrix(y_test, y_pred_optimal)
report_optimal = classification_report(y_test, y_pred_optimal, target_names=target_names)

print(f"\nAccuracy on unseen test set (Optimal K): {accuracy_optimal:.4f}\n")
print("\nConfusion Matrix (Optimal K):\n")
print(cm_optimal)

# Visualize the Confusion Matrix for the optimal model
plt.figure(figsize=(8, 6))
sns.heatmap(cm_optimal, annot=True, fmt="d",
            cmap="Greens", cbar=False,
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title(f"Confusion Matrix for Optimal KNN (K={best_knn_model.n_neighbors})",
          fontsize=14)
plt.show()

print("\nClassification Report (Optimal K): \n")
print(report_optimal)

print("\nOptimal Model Evaluation Complete!\n")