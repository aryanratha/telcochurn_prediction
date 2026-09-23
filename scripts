"""
CUSTOMER CHURN PREDICTION & RETENTION INSIGHTS
================================================
Dataset: Telco Customer Churn (download from Kaggle, place CSV in same folder)
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)

sns.set_style("whitegrid")

# ---------------------------------------------------------
# 1. LOAD & CLEAN DATA
# ---------------------------------------------------------
df = pd.read_csv(r"C:\Users\DELL\OneDrive\Documents\Data Analytics Projects\churn_project\WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Shape:", df.shape)
print(df.head())
print(df.info())

# TotalCharges is loaded as object due to blank strings — fix it
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("\nMissing values after conversion:\n", df.isnull().sum()[df.isnull().sum() > 0])

# These blanks are new customers (tenure = 0), so fill with 0 instead of dropping
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# Drop customerID — it's an identifier, not a feature
df.drop("customerID", axis=1, inplace=True)

# Target variable to binary
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

print(f"\nOverall churn rate: {df['Churn'].mean()*100:.2f}%")

# ---------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------

# 2a. Churn by contract type
plt.figure(figsize=(7, 4))
sns.barplot(data=df, x="Contract", y="Churn", errorbar=None)
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn Rate")
plt.tight_layout()
plt.savefig("churn_by_contract.png")
plt.show()

# 2b. Churn by tenure
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=30)
plt.title("Tenure Distribution by Churn Status")
plt.tight_layout()
plt.savefig("churn_by_tenure.png")
plt.show()

# 2c. Churn by monthly charges
plt.figure(figsize=(7, 4))
sns.boxplot(data=df, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Churn")
plt.tight_layout()
plt.savefig("churn_by_charges.png")
plt.show()

# 2d. Churn by internet service type
plt.figure(figsize=(7, 4))
sns.barplot(data=df, x="InternetService", y="Churn", errorbar=None)
plt.title("Churn Rate by Internet Service Type")
plt.tight_layout()
plt.savefig("churn_by_internet.png")
plt.show()

# ---------------------------------------------------------
# 3. FEATURE ENGINEERING
# ---------------------------------------------------------
df_model = df.copy()

# Encode binary Yes/No columns
binary_cols = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
for col in binary_cols:
    df_model[col] = df_model[col].map({"Yes": 1, "No": 0})

df_model["gender"] = df_model["gender"].map({"Male": 1, "Female": 0})

# One-hot encode multi-category columns
multi_cat_cols = ["MultipleLines", "InternetService", "OnlineSecurity",
                  "OnlineBackup", "DeviceProtection", "TechSupport",
                  "StreamingTV", "StreamingMovies", "Contract",
                  "PaymentMethod"]
df_model = pd.get_dummies(df_model, columns=multi_cat_cols, drop_first=True)

X = df_model.drop("Churn", axis=1)
y = df_model["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale numeric features
scaler = StandardScaler()
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# ---------------------------------------------------------
# 4. MODEL 1 — LOGISTIC REGRESSION 
# ---------------------------------------------------------
log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)
log_probs = log_model.predict_proba(X_test)[:, 1]

print("\n--- Logistic Regression ---")
print(classification_report(y_test, log_preds))
print("ROC-AUC:", roc_auc_score(y_test, log_probs))

# ---------------------------------------------------------
# 5. MODEL 2 — RANDOM FOREST
# ---------------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

print("\n--- Random Forest ---")
print(classification_report(y_test, rf_preds))
print("ROC-AUC:", roc_auc_score(y_test, rf_probs))

# ---------------------------------------------------------
# 6. MODEL COMPARISON TABLE
# ---------------------------------------------------------
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [accuracy_score(y_test, log_preds), accuracy_score(y_test, rf_preds)],
    "Precision": [precision_score(y_test, log_preds), precision_score(y_test, rf_preds)],
    "Recall": [recall_score(y_test, log_preds), recall_score(y_test, rf_preds)],
    "F1": [f1_score(y_test, log_preds), f1_score(y_test, rf_preds)],
    "ROC-AUC": [roc_auc_score(y_test, log_probs), roc_auc_score(y_test, rf_probs)],
})
print("\nModel Comparison:\n", results)
results.to_csv("model_comparison.csv", index=False)

# ROC curve comparison plot
plt.figure(figsize=(6, 5))
for name, probs in [("Logistic Regression", log_probs), ("Random Forest", rf_probs)]:
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, probs):.2f})")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("roc_comparison.png")
plt.show()

# ---------------------------------------------------------
# 7. FEATURE IMPORTANCE
# ---------------------------------------------------------
importances = pd.Series(rf_model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 10 Features Driving Churn (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

print("\nTop churn drivers:\n", top_features)

top_features.to_csv("feature_importance.csv", header=["Importance"])

# ---------------------------------------------------------
# 8. EXPORT PREDICTIONS FOR POWER BI
# ---------------------------------------------------------
export_df = X_test.copy()
export_df["ActualChurn"] = y_test
export_df["PredictedChurn"] = rf_preds
export_df["ChurnProbability"] = rf_probs

# Bring back the original index as a pseudo-CustomerID for joining
export_df.reset_index(inplace=True)
export_df.rename(columns={"index": "CustomerID"}, inplace=True)

export_df.to_csv("churn_predictions_export.csv", index=False)
print("\nExported predictions to churn_predictions_export.csv")
