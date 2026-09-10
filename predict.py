import joblib
import numpy as np
import pandas as pd

# 1. Load the trained artifacts
model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# 2. Grab an actual sample transaction from the dataset to test
df = pd.read_csv("data/creditcard.csv")

# Pick one legitimate sample (Class 0) and one fraudulent sample (Class 1)
legit_sample = df[df["Class"] == 0].iloc[0]
fraud_sample = df[df["Class"] == 1].iloc[0]

def test_transaction(sample, label):
    # Scale Amount and Time using the fitted scaler
    scaled_amount = scaler.transform([[sample["Amount"]]])[0][0]
    scaled_time = scaler.transform([[sample["Time"]]])[0][0]

    # Extract V1 to V28
    features = sample.drop(["Time", "Amount", "Class"]).values.tolist()

    # Match exact feature order: V1..V28, scaled_amount, scaled_time
    full_vector = np.array([features + [scaled_amount, scaled_time]])

    prediction = model.predict(full_vector)[0]
    probability = model.predict_proba(full_vector)[0][1]

    print(f"--- Testing {label} Transaction ---")
    print(f"Actual Class: {int(sample['Class'])}")
    print(f"Predicted: {'🚨 FRAUD' if prediction == 1 else '✅ NORMAL'}")
    print(f"Fraud Probability: {probability:.2%}\n")

test_transaction(legit_sample, "Known Legitimate")
test_transaction(fraud_sample, "Known Fraud")