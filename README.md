# Student Performance Predictor

A small ML web Application that predicts a student is pass or fail based on academic performance.

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
- Pandas:- working with tables and datasets.
- NumPy: - working with numerical calculations and arrays.
- Scikit-learn: - machine learning library for KNN, train-test split.
- Matplotlib: - working for graphs and visualizations.
- Streamlit: - framework for web application on python based.
- Joblib: - for parallel computing, and serialization.

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