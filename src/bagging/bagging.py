import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
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

# === Define Bagging Classifier ===
bagging_model = BaggingClassifier(
    estimator=LogisticRegression(class_weight="balanced", max_iter=1000),
    n_estimators=50,
    max_samples=0.8,
    max_features=1.0,
    bootstrap=True,
    n_jobs=-1,
    random_state=42
)

# === Fit the model ===
bagging_model.fit(X_train_scaled, y_train)

# === Evaluate ===
y_pred = bagging_model.predict(X_test_scaled)
y_proba = bagging_model.predict_proba(X_test_scaled)[:, 1]

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (BaggingClassifier)")
plt.tight_layout()
plt.savefig("bagging_confusion_matrix.png", dpi=300)

# === Classification Report ===
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.rename(index={"0": "Legit(0)", "1": "Fraud(1)"}, inplace=True)
report_df.to_csv("bagging_classification_report.csv", index=True)

# === ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (BaggingClassifier)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("bagging_roc_curve.png", dpi=300)

print(f"Test ROC AUC Score: {roc_auc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))
