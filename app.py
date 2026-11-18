import streamlit as st
import pandas as pd
import joblib

# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="centered"
)

# Load the saved model pipeline
model = joblib.load('model.joblib')

# App Header
st.title("🎓 Student Performance Predictor")
st.markdown("Provide the student's study habits below to predict their exam outcome.")
st.divider()

# Input Fields organized in two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
    hours_studied = st.number_input("Hours Studied", min_value=0.0, max_value=50.0, value=10.0, step=0.5)

with col2:
    practice_tests = st.number_input("Practice Tests Taken", min_value=0, max_value=20, value=2, step=1)
    sleep_hours = st.number_input("Sleep Hours (Daily)", min_value=0.0, max_value=24.0, value=7.0, step=0.5)


study_method = st.selectbox("Primary Study Method", ["Self-study", "Group", "Video"])

st.markdown("<br>", unsafe_allow_html=True)

# predict button
if st.button("Predict Outcome", type="primary", use_container_width=True):
    # Creating input dataframe matching model training columns
    input_data = pd.DataFrame([{
        'attendance': attendance,
        'hours_studied': hours_studied,
        'practice_tests': practice_tests,
        'sleep_hours': sleep_hours,
        'study_method': study_method
    }])

    # Prediction using the loaded pipeline
    prediction = model.predict(input_data)[0]

    st.divider()

    if prediction == 1:
        st.success("### Result: The student is likely to **PASS**! 🎉")
    else:
        st.error("### Result: The student is likely to **FAIL**. ⚠️")