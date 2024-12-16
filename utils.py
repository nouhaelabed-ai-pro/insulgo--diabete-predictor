import numpy as np
import joblib

def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('model/diabetes_model.pkl')
        scaler = joblib.load('model/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

def preprocess_input(data, scaler):
    """Preprocess input data using the saved scaler"""
    if not scaler:
        return None
    
    try:
        # Convert input data to numpy array and reshape
        data_array = np.array(data).reshape(1, -1)
        # Scale the data
        scaled_data = scaler.transform(data_array)
        return scaled_data
    except Exception as e:
        print(f"Error preprocessing input: {str(e)}")
        return None

def validate_input(data):
    """Validate input data"""
    required_fields = [
        'pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
        'insulin', 'bmi', 'diabetes_pedigree', 'age'
    ]
    
    # Check if all required fields are present
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
        
        # Check if values are numeric and non-negative
        try:
            value = float(data[field])
            if value < 0:
                return False, f"Field {field} cannot be negative"
        except ValueError:
            return False, f"Field {field} must be a number"
    
    return True, "Input data is valid"

def format_prediction_result(prediction, probability):
    """Format the prediction result"""
    return {
        'prediction': bool(prediction),
        'probability': float(probability),
        'risk_level': 'High' if prediction else 'Low',
        'confidence': f"{probability * 100:.1f}%"
    }
