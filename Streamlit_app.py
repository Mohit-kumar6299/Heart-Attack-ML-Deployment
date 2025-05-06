import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('extra_trees_model.pkl', 'rb'))

st.title("Heart Disease Prediction App")

# Input fields
Age = st.number_input("Age", min_value=0.0, format="%.2f")
Cholesterol = st.number_input("Cholesterol", min_value=0.0, format="%.2f")
Heart_rate = st.number_input("Heart rate", min_value=0.0, format="%.2f")
Diabetes = st.number_input("Diabetes", min_value=0.0, format="%.2f")
Family_History = st.number_input("Family History", min_value=0.0, format="%.2f")
Smoking = st.number_input("Smoking", min_value=0.0, format="%.2f")
Obesity = st.number_input("Obesity", min_value=0.0, format="%.2f")
Alcohol_Consumption = st.number_input("Alcohol Consumption", min_value=0.0, format="%.2f")
Exercise_Hours_Per_Week = st.number_input("Exercise Hours Per Week", min_value=0.0, format="%.2f")
Diet = st.number_input("Diet", min_value=0.0, format="%.2f")
Previous_Heart_Problems = st.number_input("Previous Heart Problems", min_value=0.0, format="%.2f")
Medication_Use = st.number_input("Medication Use", min_value=0.0, format="%.2f")
Stress_Level = st.number_input("Stress Level", min_value=0.0, format="%.2f")
Sedentary_Hours_Per_Day = st.number_input("Sedentary Hours Per Day", min_value=0.0, format="%.2f")
Income = st.number_input("Income", min_value=0.0, format="%.2f")
BMI = st.number_input("BMI", min_value=0.0, format="%.2f")
Triglycerides = st.number_input("Triglycerides", min_value=0.0, format="%.2f")
Physical_Activity_Days_Per_Week = st.number_input("Physical Activity Days Per Week", min_value=0.0, format="%.2f")
Sleep_Hours_Per_Day = st.number_input("Sleep Hours Per Day", min_value=0.0, format="%.2f")
Blood_sugar = st.number_input("Blood sugar", min_value=0.0, format="%.2f")
CK_MB = st.number_input("CK-MB", min_value=0.0, format="%.2f")
Troponin = st.number_input("Troponin", min_value=0.0, format="%.2f")
Gender = st.number_input("Gender (0=Female, 1=Male)", min_value=0.0, max_value=1.0, format="%.0f")
Systolic_blood_pressure = st.number_input("Systolic blood pressure", min_value=0.0, format="%.2f")
Diastolic_blood_pressure = st.number_input("Diastolic blood pressure", min_value=0.0, format="%.2f")

# Prediction
if st.button("Predict"):
    data = np.array([[
        Age, Cholesterol, Heart_rate, Diabetes, Family_History,
        Smoking, Obesity, Alcohol_Consumption, Exercise_Hours_Per_Week,
        Diet, Previous_Heart_Problems, Medication_Use, Stress_Level,
        Sedentary_Hours_Per_Day, Income, BMI, Triglycerides,
        Physical_Activity_Days_Per_Week, Sleep_Hours_Per_Day,
        Blood_sugar, CK_MB, Troponin, Gender,
        Systolic_blood_pressure, Diastolic_blood_pressure
    ]])
    
    prediction = model.predict(data)
    
    if prediction[0] == 0:
        st.success("Prediction: No disease")
    else:
        st.error("Prediction: Disease detected")
