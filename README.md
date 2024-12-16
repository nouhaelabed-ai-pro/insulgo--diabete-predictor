# Insulgo - Diabetes Risk Prediction Model

This project generates synthetic diabetes data and trains a machine learning model to predict diabetes risk levels.

## Setup

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Generate the synthetic data:
```bash
python generate_diabetes_data.py
```

3. Train the model:
```bash
python train_model.py
```

## Project Structure

- `generate_diabetes_data.py`: Generates synthetic diabetes data with risk levels
- `train_model.py`: Trains a Random Forest model to predict risk levels
- `diabetes_data.csv`: Generated synthetic dataset
- `diabetes_risk_model.joblib`: Trained model (generated after running train_model.py)
- `scaler.joblib`: Feature scaler (generated after running train_model.py)

## Features

The dataset includes the following features:
- Age
- Gender
- BMI (Body Mass Index)
- Fasting Glucose
- HbA1c
- Blood Pressure (Systolic and Diastolic)
- Family History
- Physical Activity
- Smoking Status

Risk levels are classified as:
- Low
- Medium
- High

## Model

The project uses a Random Forest Classifier to predict risk levels based on the input features. The model is trained on 80% of the data and evaluated on the remaining 20%.
