import streamlit as st
import numpy as np
import joblib

# Load saved artifacts
model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Credit Card Fraud Detection")

st.write("Enter transaction values to predict fraud")

inputs = []



# Add Time input first (as in training data)
time = st.number_input("Time", value=0.0, format="%.10f")
inputs.append(time)

# V1 to V28
for i in range(1, 29):
    value = st.number_input(f"V{i}", value=0.0, format="%.10f")
    inputs.append(value)

# Amount
amount = st.number_input("Amount", value=0.0, format="%.10f")
inputs.append(amount)

# Order: Time, V1-V28, Amount (matches training)
X = np.array(inputs).reshape(1, -1)

if st.button("Predict"):
    # Scale all features (scaler expects 30 features)
    X_scaled = scaler.transform(X)
    prediction = model.predict(X_scaled)
    prob = model.predict_proba(X_scaled)[0][1]

    st.write(f"Predicted class: {prediction[0]}")
    st.write(f"Probability of fraud: {prob:.4f}")
    st.write(f"Scaled input: {X_scaled}")

    if prediction[0] == 1:
        st.error(f"Fraud Transaction  (Probability: {prob:.2f})")
    else:
        st.success(f"Legitimate Transaction  (Probability: {prob:.2f})")
