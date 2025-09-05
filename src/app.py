import streamlit as st
import pandas as pd
import joblib

# Load model and encoders from correct paths
model = joblib.load("models/xgb_credit_model.pkl")

# Encoders with filenames adjusted to avoid spaces in filenames
encoder_files = {
    "Sex": "models/Sex_encoder.pkl",
    "Housing": "models/Housing_encoder.pkl",
    "Saving accounts": "models/Saving_accounts_encoder.pkl",
    "Checking account": "models/Checking_account_encoder.pkl"
}
encoders = {key: joblib.load(fname) for key, fname in encoder_files.items()}

st.title("Credit Risk Prediction App")
st.write("Enter applicant information to predict credit risk")

# Input widgets
age = st.number_input("Age", min_value=18, max_value=100, value=35)
sex = st.selectbox("Sex", ["Male", "Female"])
job = st.number_input("Job (0-3)", min_value=0, max_value=10, value=2)
housing = st.selectbox("Housing", ["Rent", "Own", "Free"])
saving_account = st.selectbox("Saving Accounts", ["Little", "moderate", "rich", "quite rich"])
checking_account = st.selectbox("Checking Accounts", ["Moderate", "little", "rich"])
credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
duration = st.number_input("Duration (months)", min_value=12, max_value=72, value=12)

# Normalize categorical inputs to encoder expected format (lowercase)
def safe_transform(encoder, value):
    try:
        return encoder.transform([value.lower()])[0]
    except ValueError:
        st.error(f"Invalid input value '{value}' for encoder.")
        st.stop()

# Prepare input dataframe with transformed categorical values
input_dict = {
    "Age": age,
    "Sex": safe_transform(encoders["Sex"], sex),
    "Job": job,
    "Housing": safe_transform(encoders["Housing"], housing),
    "Saving accounts": safe_transform(encoders["Saving accounts"], saving_account),
    "Checking account": safe_transform(encoders["Checking account"], checking_account),
    "Credit amount": credit_amount,
    "Duration": duration
}

input_df = pd.DataFrame([input_dict])

# Predict button and output
if st.button("Predict Risk"):
    pred = model.predict(input_df)[0]
    if pred == 1:
        st.success("✅ The predicted credit risk is: **GOOD** (Lower Risk)")
    else:
        st.error("⚠️ The predicted credit risk is: **BAD** (Higher Risk)")

# Optional: show input summary for user confirmation
with st.expander("Show input details"):
    st.json(input_dict)
