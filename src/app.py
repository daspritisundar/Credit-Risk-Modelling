# 1 Good(lower risk) 0 Bad (Higher Risk)

import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models//xgb_credit_model.pkl")
columns = joblib.load("models//xgb_credit_model_columns.pkl")
encoders= {col: joblib.load(f"models//{col}_encoder.pkl") for col in["Sex","Housing","Saving accounts","Checking account"]}

st.title("Credit Risk Prediction App")
st.write("Enter application information to predict if the credit risk is good ")

age=st.number_input("Age",min_value=18,max_value=100)
sex=st.selectbox("Sex",["male","female"])
job=st.number_input("Job(0-3)",min_value=0,max_value=10)
housing=st.selectbox("Housing",['own', 'rent', 'free'])
saving_account=st.selectbox("Saving accounts",["little","moderate","rich","quite rich"])
checking_account=st.selectbox("Checking account",["little","moderate","rich"]) 
credit_amount=st.number_input("Credit amount",min_value=0,value=1000)
duration=st.number_input("Duration(month)",min_value=12,value=12)

input_df = input_df[columns] 

expected_cols = model.get_booster().feature_names
input_df = input_df[expected_cols] 


if st.button("Predict Risk"):
    pred=model.predict(input_df)[0]

    if pred==1:
        st.success("The predicted credit risk is:**GOOD**")
    else:
        st.error("The predicted credit risk is: **BAD**")
