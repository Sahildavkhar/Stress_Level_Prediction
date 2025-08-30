import streamlit as st
import pandas as pd
import joblib

# Load saved objects
model = joblib.load('stress_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
feature_names = joblib.load('feature_names.pkl')

# Mapping for Sleep Duration input
sleep_map = {
    "Less than 4 hours": 2,
    "4-6 hours": 5,
    "6-8 hours": 7,
    "More than 8 hours": 9
}

st.title("Stress Level Prediction")

# User inputs for reduced features
age = st.number_input("Age", min_value=10, max_value=100, value=25)
gender = st.selectbox("Gender", options=label_encoders['Gender'].classes_)
academic_pressure = st.slider("Academic Pressure (1 - 10)", 1, 10, 5)
sleep_duration = st.selectbox("Sleep Duration", options=list(sleep_map.keys()))
financial_stress = st.slider("Financial Stress (1 - 10)", 1, 10, 5)

# Prepare input DataFrame
input_dict = {
    'Age': age,
    'Gender': gender,
    'Academic Pressure': academic_pressure,
    'Sleep Duration': sleep_duration,
    'Financial Stress': financial_stress
}

input_df = pd.DataFrame([input_dict])

# Encode categorical columns using saved label encoders
for col, le in label_encoders.items():
    if col in input_df.columns:
        input_df[col] = le.transform(input_df[col].astype(str))

# Map Sleep Duration string to numeric
input_df['Sleep Duration'] = input_df['Sleep Duration'].map(sleep_map)

# Reorder columns to match training data
input_df = input_df[feature_names]

# Scale input features
input_scaled = scaler.transform(input_df)

# Prediction
if st.button("Predict Stress Level"):
    pred = model.predict(input_scaled)[0]
    st.success(f"Predicted Stress Level: {pred}")
``