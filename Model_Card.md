# Model Card: Student Performance Predictor

## Purpose
To build and demonstrate an end-to-end reproducible machine learning pipeline that predicts student academic outcomes using pre-exam study metrics.

## Input (Features)
The model takes 5 input features representing student study habits:
- **`attendance`**,  **`hours_studied`**, **`practice_tests`**, **`sleep_hours`**, **`study_method`**.

## Target 
- **`passed`**

## Evaluation
Evaluated fairly on a held-out test set using stratification:
- **Random Forest (Selected Model)**: 
  - Accuracy: ~78.5%
  - F1-Score: 0.75
  - Confusion Matrix: 26 TN, 5 FP, 7 FN, 18 TP.
- **Other Models**: Dummy Baseline (55% accuracy) and Logistic Regression (75% accuracy).

## Limitations
- **Limits**: Trained purely on synthetic classroom data, not real student records.
- **Use**: Must **not** be used to officially judge, grade, track, or penalize real students. Predictions are for demonstration purposes only.