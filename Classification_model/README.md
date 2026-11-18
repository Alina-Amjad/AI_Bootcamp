# Student Performance Prediction Project

## Overview
It's an end-to-end machine learning Project created using pipeline and choosing the best model from two(Logistic Regression and Random Forest)
to predict whether a student will pass (`passed = 1`) or fail (`passed = 0`) based on 5 features(hours studied, attendance, study method, practice tests and sleep hours).

## Project Structure
- `model_training.ipynb`: Containing training code.Loding model and making predictions on latest dataset.
- `student_learning.csv`: The training dataset.
- `new_student_records.csv`: Unseen records used for predictions.
- `model.joblib`: The Random Forest Model.
- `app.py`: Streamlit web app.

## How to Run the App
1. Install all libraries used (`pandas`, `scikit-learn`, `joblib`, `streamlit`).
2. Run the Streamlit app from your terminal:
   ```bash
   streamlit run app.py
