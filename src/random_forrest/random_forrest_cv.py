import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# === Load Data ===
df = pd.read_parquet("../../data/processed/creditcard_processed.parquet")
X = df.drop(columns=["Class"])
y = df["Class"]

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Feature Scaling (optional for tree-based models, still applied for consistency) ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Define Parameter Grid for Random Forest ===
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "class_weight": ["balanced"],
}

rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)

# === Run GridSearchCV ===
# grid_search = GridSearchCV(
#     estimator=rf_base,
#     param_grid=param_grid,
#     scoring="roc_auc",
#     cv=3,
#     verbose=2,
#     n_jobs=-1,
#     return_train_score=True
# )
random_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_grid,
    n_iter=10,  # number of combinations to try
    scoring="roc_auc",
    cv=3,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(X_train_scaled, y_train)
best_rf = random_search.best_estimator_

# === Report Best Parameters ===
print("Best Parameters:", random_search.best_params_)
print(f"Best CV ROC AUC Score: {random_search.best_score_:.4f}")

# === Save CV Results and Feature Importances ===
cv_results_df = pd.DataFrame(random_search.cv_results_)
cv_results_df.to_csv("rf_gridsearchcv_results.csv", index=False)

feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_rf.feature_importances_
}).sort_values(by="Importance", ascending=False)
feature_importances.to_csv("rf_feature_importances.csv", index=False)

# === Evaluate on Test Set ===
y_pred = best_rf.predict(X_test_scaled)
y_proba = best_rf.predict_proba(X_test_scaled)[:, 1]

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Tuned Random Forest)")
plt.tight_layout()
plt.savefig("rf_confusion_matrix_gridsearchcv.png", dpi=300)

# Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.rename(index={"0": "Legit(0)", "1": "Fraud(1)"}, inplace=True)
report_df.to_csv("rf_classification_report_gridsearchcv.csv", index=True)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Tuned Random Forest)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("rf_roc_curve_gridsearchcv.png", dpi=300)

# Final Print
print(f"Test ROC AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
