import streamlit as st
import pandas as pd
import joblib
import os

# Paths for your model and encoders
MODEL_PATH = "models/xgb_credit_model.pkl"
ENCODER_PATHS = {
    "sex": "models/Sex_encoder.pkl",
    "housing": "models/Housing_encoder.pkl",
    "saving_accounts": "models/Saving_accounts_encoder.pkl",
    "checking_account": "models/Checking_account_encoder.pkl"
}

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)

# Check and load encoders
for key, path in ENCODER_PATHS.items():
    if not os.path.exists(path):
        st.error(f"Encoder file for '{key}' not found at: {path}")
        st.stop()

encoders = {key: joblib.load(path) for key, path in ENCODER_PATHS.items()}

st.title("Credit Risk Prediction App")
st.write("Enter applicant information to predict credit risk")

# User inputs with appropriate default values
age = st.number_input("Age", min_value=18, max_value=75, value=35)
sex = st.selectbox("Sex", ["male", "female"])
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=2)
housing = st.selectbox("Housing", ["own", "free", "rent"])
saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich", "NA"])
checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich", "NA"])
credit_amount = st.number_input("Credit Amount", min_value=250, max_value=18424, value=1000)
duration = st.number_input("Duration (months)", min_value=4, max_value=72, value=12)

# Function to safely transform categorical input
def safe_transform(encoder, value):
    if value == "NA":
        try:
            return encoder.transform(["NA"])[0]
        except ValueError:
            return encoder.classes_.tolist().index("little") if "little" in encoder.classes_ else 0
    else:
        try:
            return encoder.transform([value])[0]
        except ValueError:
            st.error(f"Invalid input value '{value}' for encoder.")
            st.stop()

# Prepare data for prediction
input_data = {
    "age": age,
    "sex": safe_transform(encoders["sex"], sex),
    "job": job,
    "housing": safe_transform(encoders["housing"], housing),
    "saving_accounts": safe_transform(encoders["saving_accounts"], saving_accounts),
    "checking_account": safe_transform(encoders["checking_account"], checking_account),
    "credit_amount": credit_amount,
    "duration": duration
}

input_df = pd.DataFrame([input_data])

# Ensure input columns match model features
expected_features = model.get_booster().feature_names
input_df = input_df[expected_features]

if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]
    if pred == 1:
        st.success("✅ The predicted credit risk is: GOOD (Low Risk)")
    else:
        st.error("⚠️ The predicted credit risk is: BAD (High Risk)")

with st.expander("Show input details"):
    st.json(input_data)
