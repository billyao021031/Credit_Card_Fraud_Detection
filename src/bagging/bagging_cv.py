import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
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

# === Feature Scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Define Base Estimator ===
base_estimator = LogisticRegression(
    class_weight="balanced",
    solver="liblinear",
    max_iter=1000
)

# === Define Bagging Classifier and Param Grid ===
bagging_model = BaggingClassifier(
    estimator=base_estimator,
    random_state=42,
    n_jobs=-1
)

param_dist = {
    "n_estimators": [10, 25, 50, 75, 100],
    "max_samples": [0.5, 0.7, 0.9, 1.0],
    "max_features": [0.5, 0.7, 1.0],
    "bootstrap": [True, False]
}

# === RandomizedSearchCV ===
random_search = RandomizedSearchCV(
    estimator=bagging_model,
    param_distributions=param_dist,
    n_iter=10,
    scoring="roc_auc",
    cv=3,
    verbose=2,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train_scaled, y_train)
best_bagging = random_search.best_estimator_

# === Save CV Results ===
cv_results_df = pd.DataFrame(random_search.cv_results_)
cv_results_df.to_csv("bagging_randomsearchcv_results.csv", index=False)

# === Evaluation on Test Set ===
y_pred = best_bagging.predict(X_test_scaled)
y_proba = best_bagging.predict_proba(X_test_scaled)[:, 1]

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Tuned Bagging)")
plt.tight_layout()
plt.savefig("bagging_confusion_matrix_tuned.png", dpi=300)

# === Classification Report ===
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.rename(index={"0": "Legit(0)", "1": "Fraud(1)"}, inplace=True)
report_df.to_csv("bagging_classification_report_tuned.csv", index=True)

# === ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Tuned Bagging)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("bagging_roc_curve_tuned.png", dpi=300)

# === Final Print ===
print("Best Parameters:", random_search.best_params_)
print(f"Best CV ROC AUC Score: {random_search.best_score_:.4f}")
print(f"Test ROC AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
