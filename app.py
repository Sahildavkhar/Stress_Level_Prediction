import streamlit as st
import pandas as pd
import joblib

# Load saved model and preprocessing objects
model = joblib.load('stress_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')
feature_names = joblib.load('feature_names.pkl')

# Mapping for Sleep Duration input (same as used in model training)
sleep_map = {
    "Less than 5 hours": 2,
    "5-6 hours": 5,
    "6-8 hours": 7,
    "More than 8 hours": 9
}

# Mapping predicted class to readable text
stress_labels = {
    0: "Low",
    1: "Moderate",
    2: "High",
    3: "Very High"
}

st.title("🧠 Stress Level Prediction")

st.write("Enter your details below to predict your stress level based on lifestyle and personal factors.")

# User inputs for the selected features
age = st.number_input("Age", min_value=10, max_value=100, value=22)
gender = st.selectbox("Gender", options=label_encoders['Gender'].classes_)
work_pressure = st.slider("Work Pressure (1 - 10)", 1, 10, 5)
sleep_duration = st.selectbox("Sleep Duration", options=list(sleep_map.keys()))
financial_stress = st.slider("Financial Stress (1 - 10)", 1, 10, 5)

# Prepare input dictionary
input_dict = {
    'Age': age,
    'Gender': gender,
    'Work Pressure': work_pressure,
    'Sleep Duration': sleep_duration,
    'Financial Stress': financial_stress
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Encode categorical columns
for col, le in label_encoders.items():
    if col in input_df.columns:
        input_df[col] = le.transform(input_df[col].astype(str))

# Map Sleep Duration to numeric
input_df['Sleep Duration'] = input_df['Sleep Duration'].map(sleep_map)

# Ensure order matches training
input_df = input_df[feature_names]

# Scale input
input_scaled = scaler.transform(input_df)

# Prediction button
if st.button("🔍 Predict Stress Level"):
    pred = model.predict(input_scaled)[0]
    st.success(f"Predicted Stress Level: **{stress_labels[pred]}** ({pred})")
    # st.info("Stress Level Scale: 0 = Low | 1 = Moderate | 2 = High | 3 = Very High")
