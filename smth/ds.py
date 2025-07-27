# fraud_detection_analysis.py
# A single-file script for the INSAID Data Science Internship Task

# --- SECTION 1: SETUP AND LIBRARIES ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings

warnings.filterwarnings('ignore')
print("--- Fraud Detection Analysis Script ---")


# --- SECTION 2: DATA LOADING AND INITIAL INSPECTION ---
print("\n[INFO] Section 2: Loading and Inspecting Data...")
try:
    df = pd.read_csv('PS_20174392719_1491204439457_log.csv')
    print("Dataset loaded successfully.")
    print("\nData Head:")
    print(df.head())
    print("\nData Info:")
    df.info()

    # Checking for missing values
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("Observation: The dataset is complete with no missing values.")

except FileNotFoundError:
    print("\n[ERROR] Dataset file 'PS_20174392719_1491204439457_log.csv' not found.")
    print("Please ensure the data file is in the same directory as this script.")
    exit()


# --- SECTION 3: DATA CLEANING AND FEATURE ENGINEERING ---
print("\n[INFO] Section 3: Data Cleaning and Feature Engineering...")

# Assessing multicollinearity
numerical_cols = df.select_dtypes(include=np.number).columns
correlation_matrix = df[numerical_cols].corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()
print("Observation: High multicollinearity noted between account balance columns, as expected.")

# Fraud occurs only in 'TRANSFER' and 'CASH_OUT' types. We filter for these.
df_filtered = df[(df['type'] == 'TRANSFER') | (df['type'] == 'CASH_OUT')].copy()

# Feature Engineering: Creating new features to capture balance discrepancies
df_filtered['errorBalanceOrig'] = df_filtered['newbalanceOrig'] + df_filtered['amount'] - df_filtered['oldbalanceOrg']
df_filtered['errorBalanceDest'] = df_filtered['oldbalanceDest'] + df_filtered['amount'] - df_filtered['newbalanceDest']

# One-Hot Encode the 'type' column for model compatibility
df_filtered = pd.get_dummies(df_filtered, columns=['type'], prefix='type', drop_first=True)
print("Feature engineering complete. New features 'errorBalanceOrig' and 'errorBalanceDest' created.")


# --- SECTION 4: VARIABLE SELECTION AND DATA SPLITTING ---
print("\n[INFO] Section 4: Selecting Variables and Splitting Data...")

# Dropping identifiers and non-predictive features
X = df_filtered.drop(['isFraud', 'nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
y = df_filtered['isFraud']
print("Features selected for the model:", list(X.columns))

# Splitting data into calibration (train) and validation (test) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Data split into training set ({X_train.shape[0]} rows) and testing set ({X_test.shape[0]} rows).")


# --- SECTION 5: MODEL DEVELOPMENT AND TRAINING ---
print("\n[INFO] Section 5: Developing and Training the Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("Model training complete.")


# --- SECTION 6: MODEL PERFORMANCE DEMONSTRATION ---
print("\n[INFO] Section 6: Evaluating Model Performance...")

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\nAUC-ROC Score: {auc_score:.4f}")
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f'Random Forest (area = {auc_score:.2f})')
plt.plot([0, 1], [0, 1],'r--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()


# --- SECTION 7: IDENTIFYING KEY PREDICTORS OF FRAUD ---
print("\n[INFO] Section 7: Identifying Key Predictive Factors...")
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

print("\nTop 5 Key Factors Predicting Fraud:")
print(feature_importance_df.head())

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Key Factors Predicting Fraud')
plt.show()


# --- SECTION 8: CONCLUSION AND RECOMMENDATIONS ---
print("\n[INFO] Section 8: Insights and Recommendations")

print("\nQuestion: Do these factors make sense? If yes, How? If not, How not?")
print("Answer: Yes, the factors make perfect sense. Fraudsters often try to empty an account, so a transaction 'amount' that is close to the 'oldbalanceOrg' is a major red flag. Similarly, discrepancies in balance updates ('errorBalanceDest') clearly indicate illegitimate activity.")

print("\nQuestion: What kind of prevention should be adopted while company update its infrastructure?")
print("""Answer:
1.  Real-Time Model Integration: Embed this model to score transactions live.
2.  Dynamic Rule-Based Alerts: Flag transactions based on the top predictors (e.g., high `amount` from a high `oldbalanceOrg` account).
3.  MFA Triggers: Automatically require multi-factor authentication for high-risk transactions.
""")

print("\nQuestion: Assuming these actions have been implemented, how would you determine if they work?")
print("""Answer:
1.  A/B Testing: Compare fraud rates between a group using the new system and a control group.
2.  KPI Monitoring: Continuously track metrics like fraud rate, false positive rate, and total financial losses to measure impact.
""")

print("\n--- End of Analysis ---")