import streamlit as st
import pandas as pd
import joblib


# Load model and scaler
model = joblib.load("model/knn_model.pkl")
scaler = joblib.load("model/scaler.pkl")


# Page configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    layout="centered"
)


# Title
st.title("Student Performance Predictor")
st.write("Predict student performance using a KNN Machine Learning model.")

st.divider()


# Input section
st.subheader("📋 Student Information")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=6.0,
    step=0.5
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=80.0,
    step=1.0
)

previous_marks = st.number_input(
    "Previous Marks",
    min_value=0.0,
    max_value=100.0,
    value=70.0,
    step=1.0
)

assignment_score = st.number_input(
    "Assignment Score",
    min_value=0.0,
    max_value=100.0,
    value=75.0,
    step=1.0
)


st.divider()


# Prediction
if st.button("Predict Performance", use_container_width=True):

    # Create input DataFrame
    new_student = pd.DataFrame([{
        "study_hours": study_hours,
        "attendance": attendance,
        "previous_marks": previous_marks,
        "assignment_score": assignment_score
    }])

    # Scale input
    new_student_scaled = scaler.transform(new_student)

    # Prediction
    prediction = model.predict(new_student_scaled)[0]

    # Probability
    probabilities = model.predict_proba(new_student_scaled)[0]

    fail_probability = probabilities[0] * 100
    pass_probability = probabilities[1] * 100


    # Display result
    st.subheader("Prediction Result")

    if prediction == 1:

        st.success("Student is likely to PASS")

    else:

        st.error("Student is likely to FAIL")


    # Probability columns
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Pass Probability",
            f"{pass_probability:.2f}%"
        )

    with col2:
        st.metric(
            "Fail Probability",
            f"{fail_probability:.2f}%"
        )


    # Input summary
    st.subheader("Student Data")

    st.dataframe(
        new_student,
        use_container_width=True
    )


# Footer
st.divider()

st.caption(
    "Built using Python, Pandas, Scikit-learn, KNN and Streamlit"
)