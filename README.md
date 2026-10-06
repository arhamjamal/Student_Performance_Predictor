# 🎓 Student Performance Predictor

A machine learning web application that predicts whether a student is likely to Pass or Fail based on academic performance.

## Features

- Study Hours
- Attendance
- Previous Marks
- Assignment Score
- KNN Classification
- PASS/FAIL prediction
- Prediction probability
- Interactive Streamlit interface

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib

## Machine Learning

The project uses:

- StandardScaler for feature scaling
- K-Nearest Neighbors (KNN) for classification
- 80/20 train-test split
- Accuracy, classification report and confusion matrix for evaluation

## Project Structure

Student-Performance_Predictor/

├── data/
│   └── students.csv
├── model/
│   ├── train_model.py
│   ├── predict.py
│   ├── knn_model.pkl
│   └── scaler.pkl
├── analysis.py
├── app.py
└── requirements.txt

## Run the Project

Create and activate the virtual environment:

```bash
python -m venv .venv