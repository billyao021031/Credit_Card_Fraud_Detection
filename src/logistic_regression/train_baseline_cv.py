import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score

import matplotlib.pyplot as plt
import seaborn as sns

# === Load Data ===
df = pd.read_parquet("../../data/processed/creditcard_processed.parquet")
X = df.drop(columns=["Class"])
y = df["Class"]

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Scale Features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Define Model and Parameter Grid ===
param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"],
}

base_model = LogisticRegression(class_weight="balanced", max_iter=1000)

# === Run GridSearchCV ===
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=5,
    n_jobs=-1,
    verbose=2,
    return_train_score=True
)

grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# === Report Best Parameters ===
print("Best Parameters:", grid_search.best_params_)
print(f"Best Cross-Validated ROC AUC: {grid_search.best_score_:.4f}")

# === Evaluate on Test Set ===
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (GridSearchCV Optimized Model)")
plt.tight_layout()
plt.savefig("confusion_matrix_gridsearchcv.png", dpi=300, bbox_inches="tight")

# === Classification Report ===
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
report_df.rename(index={"0": "Legit(0)", "1": "Fraud(1)"}, inplace=True)
report_df.to_csv("classification_report_table_gridsearchcv.csv", index=True)

# === Save Grid Search Results === 
results_df = pd.DataFrame(grid_search.cv_results_)
results_df.to_csv("gridsearchcv_results.csv", index=False)

# === Save Feature Importances ===
coef_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": best_model.coef_[0]
}).sort_values(by="Coefficient", key=abs, ascending=False)
coef_df.to_csv("logreg_coefficients.csv", index=False)

# === ROC Curve and AUC ===
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
print("ROC AUC Score (Test Set):", roc_auc)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve (GridSearchCV Optimized Model)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve_gridsearchcv.png", dpi=300, bbox_inches="tight")

# === Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_proba)
avg_prec = average_precision_score(y_test, y_proba)

plt.figure(figsize=(7, 6))
plt.plot(recall, precision, label=f"PR Curve (AP = {avg_prec:.4f})", linewidth=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("pr_curve_gridsearchcv.png", dpi=300)
