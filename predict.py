import joblib
import pandas as pd
import numpy as np
import os
from src.model_utils import load_saved_model

def make_prediction(patient_data, model_type='logistic_regression'):
    """
    Takes in patient data, scales it using the saved scaler, 
    and predicts the outcome using the specified saved model.
    """
    scaler_path = os.path.join('models', 'scaler.joblib')
    model_path = os.path.join('models', f'{model_type}_model.joblib')
    
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        print("Error: Models and scaler not found. Please run main.py first to train the models.")
        return
        
    # Load saved models
    scaler = joblib.load(scaler_path)
    model = load_saved_model(model_path)
    
    # Preprocess patient data
    patient_df = pd.DataFrame([patient_data])
    scaled_data = scaler.transform(patient_df)
    
    # Predict
    prediction = model.predict(scaled_data)
    probability = model.predict_proba(scaled_data)[0][1] if hasattr(model, "predict_proba") else None
    
    print("\n--- Prediction Results ---")
    print(f"Model used: {model_type.replace('_', ' ').title()}")
    print(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}")
    
    if probability is not None:
        print(f"Probability of being Diabetic: {probability:.2%}")

if __name__ == "__main__":
    # Sample new patient data based on the features:
    # Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    sample_patient = {
        'Pregnancies': 2,
        'Glucose': 120,
        'BloodPressure': 75,
        'SkinThickness': 25,
        'Insulin': 100,
        'BMI': 28.5,
        'DiabetesPedigreeFunction': 0.45,
        'Age': 33
    }
    
    print("Making prediction for new patient data:")
    print(sample_patient)
    make_prediction(sample_patient, model_type='random_forest')
