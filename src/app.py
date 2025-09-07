import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load model
model = joblib.load("models/xgb_credit_model.pkl")

# Load encoders (only those you saved)
encoder_cols = ["Sex", "Housing", "Saving accounts", "Checking account"]
encoders = {}
for col in encoder_cols:
    try:
        encoders[col] = joblib.load(f"models/{col}_encoder.pkl")
    except:
        st.warning(f"⚠️ Encoder file missing for {col}, raw input will be used.")
        encoders[col] = None

# Handle Purpose separately (since you don’t have Purpose_encoder.pkl)
purpose_categories = [
    "radio/TV", "education", "furniture/equipment", "car",
    "business", "domestic appliances", "repairs", "vacation/others"
]
purpose_encoder = LabelEncoder()
purpose_encoder.fit(purpose_categories)

# Streamlit UI
st.set_page_config(page_title="Credit Risk Prediction", page_icon="💳", layout="centered")
st.title("💳 Credit Risk Prediction App")
st.write("Enter applicant details to predict whether the credit risk is **Good** or **Bad**.")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", encoders["Sex"].classes_ if encoders["Sex"] else ["male", "female"])
    job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
    housing = st.selectbox("Housing", encoders["Housing"].classes_ if encoders["Housing"] else ["Rent", "Own", "Free"])

with col2:
    saving_account = st.selectbox(
        "Saving Accounts",
        encoders["Saving accounts"].classes_ if encoders["Saving accounts"] else ["little", "moderate", "rich", "quite rich"]
    )
    checking_account = st.selectbox(
        "Checking Account",
        encoders["Checking account"].classes_ if encoders["Checking account"] else ["little", "moderate", "rich"]
    )
    credit_amount = st.number_input("Credit Amount", min_value=0, value=1000, step=100)
    duration = st.number_input("Duration (months)", min_value=4, max_value=72, value=12)

purpose = st.selectbox("Purpose", purpose_categories)

# Build input dataframe
try:
    input_df = pd.DataFrame({
        "Age": [age],
        "Sex": [encoders["Sex"].transform([sex])[0] if encoders["Sex"] else sex],
        "Job": [job],
        "Housing": [encoders["Housing"].transform([housing])[0] if encoders["Housing"] else housing],
        "Saving accounts": [encoders["Saving accounts"].transform([saving_account])[0] if encoders["Saving accounts"] else saving_account],
        "Checking account": [encoders["Checking account"].transform([checking_account])[0] if encoders["Checking account"] else checking_account],
        "Credit amount": [credit_amount],
        "Duration": [duration],
        "Purpose": [purpose_encoder.transform([purpose])[0]]
    })
except Exception as e:
    st.error(f"Encoding error: {e}")
    st.stop()

# Prediction
if st.button("🔮 Predict Risk"):
    try:
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0]

        if pred == 1:
            st.success(f"✅ The predicted credit risk is: **GOOD** (Confidence: {prob[1]*100:.2f}%)")
        else:
            st.error(f"⚠️ The predicted credit risk is: **BAD** (Confidence: {prob[0]*100:.2f}%)")

    except Exception as e:
        st.error(f"Prediction error: {e}")
