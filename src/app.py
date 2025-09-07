import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("models/xgb_credit_model.pkl")

# Load encoders
encoder_cols = ["Sex", "Housing", "Saving accounts", "Checking accounts"]
encoders = {col: joblib.load(f"models/{col}_encoder.pkl") for col in encoder_cols}

# Streamlit UI
st.set_page_config(page_title="Credit Risk Prediction", page_icon="💳", layout="centered")

st.title("💳 Credit Risk Prediction App")
st.markdown("Fill in the applicant details to predict whether the credit risk is **Good** or **Bad**.")

# Input fields
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", encoders["Sex"].classes_)
    job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
    housing = st.selectbox("Housing", encoders["Housing"].classes_)

with col2:
    saving_account = st.selectbox("Saving Accounts", encoders["Saving accounts"].classes_)
    checking_account = st.selectbox("Checking Accounts", encoders["Checking accounts"].classes_)
    credit_amount = st.number_input("Credit Amount", min_value=0, value=1000, step=100)
    duration = st.number_input("Duration (months)", min_value=4, max_value=72, value=12)

# Prepare input dataframe
try:
    input_df = pd.DataFrame({
        "Age": [age],
        "Sex": [encoders["Sex"].transform([sex])[0]],
        "Job": [job],
        "Housing": [encoders["Housing"].transform([housing])[0]],
        "Saving accounts": [encoders["Saving accounts"].transform([saving_account])[0]],
        "Checking accounts": [encoders["Checking accounts"].transform([checking_account])[0]],
        "Credit amount": [credit_amount],
        "Duration": [duration]
    })
except Exception as e:
    st.error(f"Encoding error: {e}")
    st.stop()

# Prediction button
if st.button("🔮 Predict Risk"):
    try:
        pred = model.predict(input_df)[0]

        if pred == 1:
            st.success("✅ The predicted credit risk is: **GOOD**")
        else:
            st.error("⚠️ The predicted credit risk is: **BAD**")

    except Exception as e:
        st.error(f"Prediction error: {e}")
