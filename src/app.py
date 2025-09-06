# 1 Good(lower risk) 0 Bad (Higher Risk)

import streamlit as st
import pandas as pd
import joblib

model=joblib.load("models//xgb_credit_model.pkl")
encoders= {col: joblib.load(f"{col}_encoder.okl") for col in["Sex","Housing","Saving accounts","Checking account"]}

st.title("Credit Risk Prediction App")
st.write("Enter application information to predict if the credit risk is good ")

age=st.number_input("Age",min_value=18,max_value=100)
sex=st.selectbox("Sex",["male","female"])
job=st.number_input("Job(0-3)",min_value=0,max_value=10)
housing=st.selectbox("Housing",["Rent","Own","Free"])
saving_account=st.selectbox("Saving Accounts",["Little","moderate","rich","quite rich"])
checking_account=st.selectbox("Checking Accounts",["Moderate","little","rich"]) 
credit_amount=st.number_input("Credit Amount",min_value=0,value=1000)
duration=st.number_input("Duration(month)",min_value=12,value=12)

input_df=pd.DataFrame({
    "Age":[age],
    "Sex":[encoders["Sex"].transform([sex])[0]],
    "Job":[job],
    "Housing":[encoders["Housing"].transform([housing])[0]],
    "Saving accounts":[encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking accounts": [encoders["Checking accounts"].transform([checking_accounts])[0]],
    "Credit amount":[credit_amount],
    "Duration":[duration]
})

if st.button("Predict Risk"):
    pred=model.predict(input_df)[0]

    if pred==1:
        st.success("The predicted credit risk is:**GOOD**")
    else:
        st.error("The predicted credit risk is: **BAD**")
        
