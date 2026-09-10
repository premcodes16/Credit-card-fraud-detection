import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load data
print("Loading data...")
df = pd.read_csv("data/creditcard.csv")

# 2. Scale 'Amount' and 'Time' (the V1-V28 columns are already PCA scaled)
scaler = RobustScaler()
df["scaled_amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
df["scaled_time"] = scaler.fit_transform(df["Time"].values.reshape(-1, 1))

# Drop raw unscaled columns
df.drop(["Time", "Amount"], axis=1, inplace=True)

# 3. Separate features (X) and target label (y)
X = df.drop("Class", axis=1)
y = df["Class"]

# 4. Train-Test Split (80% training, 20% testing)
# 'stratify=y' ensures both train and test get the exact same fraud ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Original Training shape: {X_train.shape}")
print(f"Normal cases in train: {sum(y_train == 0)}, Fraud cases: {sum(y_train == 1)}")

# 5. Handle Imbalance with SMOTE (ONLY apply to training data, never test data!)
print("\nApplying SMOTE to balance fraud cases...")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"Balanced Training shape: {X_train_res.shape}")
print(f"Fraud cases after SMOTE: {sum(y_train_res == 1)}")

# 6. Train the ML Model
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_res, y_train_res)

# 7. Evaluate on clean, unseen Test Data
print("\nEvaluating model on unseen test set...")
y_pred = model.predict(X_test)

print("\n" + "=" * 50)
print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))
print("=" * 50)

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))
print("=" * 50)


import joblib
#8. Save the trained model and the scaler to the 'models' folder
joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\nModel and scaler successfully saved in 'models/' folder!")