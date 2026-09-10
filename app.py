import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Credit Card Fraud Detector", layout="centered")

st.title("💳 Credit Card Fraud Detection")
st.write("Test transaction data against the trained Logistic Regression + SMOTE model.")

# Load saved models
@st.cache_resource
def load_artifacts():
    model = joblib.load("models/fraud_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()

# User Inputs
st.subheader("Transaction Parameters")
amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=149.62, step=1.0)
time_val = st.number_input("Time Elapsed (seconds)", min_value=0.0, value=0.0, step=1.0)

# Simulate PCA components V1-V28 (default to typical normal transaction means near 0)
st.write("---")
if st.button("Analyze Transaction", type="primary"):
    # Scale Amount and Time
    scaled_amount = scaler.transform([[amount]])[0][0]
    scaled_time = scaler.transform([[time_val]])[0][0]

    # Use baseline neutral features for demo (V1 to V28)
    features = [0.0] * 28
    input_vector = np.array([features + [scaled_amount, scaled_time]])

    prediction = model.predict(input_vector)[0]
    probability = model.predict_proba(input_vector)[0][1]

    if prediction == 1:
        st.error(f"🚨 FRAUD DETECTED! Probability: {probability:.2%}")
    else:
        st.success(f"✅ NORMAL TRANSACTION. Fraud Risk: {probability:.2%}")