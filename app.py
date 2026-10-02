import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Heart Disease Prediction")
st.markdown("This app predicts the likelihood of heart disease based on user input.")

age=st.slider("Age", min_value=1, max_value=120, value=30)
sex=st.selectbox("Sex", options=["Male", "Female"], index=0)

chestpain=st.selectbox("chest pain Type",["ATA","NAP","TA","ASY"])

resting_bp=st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)

cholesterol=st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)   
fasting_bs=st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["Yes", "No"], index=1)
resting_ecg=st.selectbox("Resting Electrocardiographic Results",["Normal","ST-T wave abnormality","Left ventricular hypertrophy"])
max_heart_rate=st.slider("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exception=st.selectbox("Exercise Induced Angina", options=["Yes", "No"], index=1)
oldpeak=st.number_input("Oldpeak (ST depression induced by exercise relative to rest)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
st_slope=st.selectbox("Slope of the peak exercise ST segment",["Upsloping","Flat","Downsloping"])


if st.button("Predict"):
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [1 if sex == "Male" else 0],
        "chestpain": [chestpain],
        "resting_bp": [resting_bp],
        "cholesterol": [cholesterol],
        "fasting_bs": [1 if fasting_bs == "Yes" else 0],
        "resting_ecg": [resting_ecg],
        "max_heart_rate": [max_heart_rate],
        "exception": [1 if exception == "Yes" else 0],
        "oldpeak": [oldpeak],
        "st_slope": [st_slope]
    })

    # Ensure the input data has the same columns as expected
    input_data = input_data.reindex(columns=expected_columns, fill_value=0)

    # Scale the input data
    scaled_input = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input)

    if prediction[0] == 1:
        st.error("The model predicts that you are at risk of heart disease.")
    else:
        st.success("The model predicts that you are not at risk of heart disease.")