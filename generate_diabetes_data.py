import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from tqdm import tqdm

# Initialize Faker
fake = Faker()

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_bp_systolic():
    """Generate systolic blood pressure in medical ranges"""
    ranges = [
        (90, 119, 0.4),    # Normal
        (120, 129, 0.2),   # Elevated
        (130, 139, 0.2),   # Stage 1 hypertension
        (140, 180, 0.2)    # Stage 2 hypertension
    ]
    
    selected_range = random.choices(ranges, weights=[r[2] for r in ranges])[0]
    return random.randint(selected_range[0], selected_range[1])

def generate_bp_diastolic():
    """Generate diastolic blood pressure in medical ranges"""
    ranges = [
        (60, 79, 0.4),     # Normal
        (80, 89, 0.3),     # Stage 1 hypertension
        (90, 110, 0.3)     # Stage 2 hypertension
    ]
    
    selected_range = random.choices(ranges, weights=[r[2] for r in ranges])[0]
    return random.randint(selected_range[0], selected_range[1])

def generate_fasting_glucose():
    """Generate fasting glucose in medical ranges"""
    ranges = [
        (70, 99, 0.4),     # Normal
        (100, 125, 0.3),   # Prediabetes
        (126, 200, 0.3)    # Diabetes
    ]
    
    selected_range = random.choices(ranges, weights=[r[2] for r in ranges])[0]
    return round(random.uniform(selected_range[0], selected_range[1]), 1)

def generate_hba1c():
    """Generate HbA1c in medical ranges"""
    ranges = [
        (4.0, 5.6, 0.4),   # Normal
        (5.7, 6.4, 0.3),   # Prediabetes
        (6.5, 12.0, 0.3)   # Diabetes
    ]
    
    selected_range = random.choices(ranges, weights=[r[2] for r in ranges])[0]
    return round(random.uniform(selected_range[0], selected_range[1]), 1)

def generate_bmi():
    """Generate BMI in medical ranges"""
    ranges = [
        (18.5, 24.9, 0.3),  # Normal
        (25.0, 29.9, 0.35), # Overweight
        (30.0, 45.0, 0.35)  # Obese
    ]
    
    selected_range = random.choices(ranges, weights=[r[2] for r in ranges])[0]
    return round(random.uniform(selected_range[0], selected_range[1]), 1)

def generate_diabetes_data(n_samples=50000):
    data = []
    
    for _ in tqdm(range(n_samples), desc="Generating diabetes data"):
        # Generate basic patient information
        age = random.randint(18, 85)
        gender = random.choice(['M', 'F'])
        
        # Generate realistic medical measurements using the new functions
        bmi = generate_bmi()
        fasting_glucose = generate_fasting_glucose()
        hba1c = generate_hba1c()
        blood_pressure_systolic = generate_bp_systolic()
        blood_pressure_diastolic = generate_bp_diastolic()
        
        # Family history and lifestyle factors
        family_history = random.choice([0, 1])
        physical_activity = random.randint(0, 7)  # days per week
        smoking = random.choice([0, 1])
        
        # Calculate risk score based on various factors
        risk_score = 0
        
        # Age factor
        if age > 45:
            risk_score += 2
        elif age > 35:
            risk_score += 1
            
        # BMI factor
        if bmi >= 30:
            risk_score += 2
        elif bmi >= 25:
            risk_score += 1
            
        # Glucose factor
        if fasting_glucose >= 126:
            risk_score += 3
        elif fasting_glucose >= 100:
            risk_score += 2
            
        # HbA1c factor
        if hba1c >= 6.5:
            risk_score += 3
        elif hba1c >= 5.7:
            risk_score += 2
            
        # Blood pressure factor
        if blood_pressure_systolic >= 140 or blood_pressure_diastolic >= 90:
            risk_score += 2
        elif blood_pressure_systolic >= 130 or blood_pressure_diastolic >= 80:
            risk_score += 1
            
        # Family history factor
        if family_history:
            risk_score += 1
            
        # Physical activity factor (inverse relationship)
        if physical_activity < 3:
            risk_score += 1
            
        # Smoking factor
        if smoking:
            risk_score += 1
            
        # Determine risk level based on total score
        if risk_score >= 10:
            risk_level = 'High'
        elif risk_score >= 6:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
            
        # Create record
        record = {
            'PatientID': fake.uuid4(),
            'Age': age,
            'Gender': gender,
            'BMI': bmi,
            'FastingGlucose': fasting_glucose,
            'HbA1c': hba1c,
            'BloodPressureSystolic': blood_pressure_systolic,
            'BloodPressureDiastolic': blood_pressure_diastolic,
            'FamilyHistory': family_history,
            'PhysicalActivity': physical_activity,
            'Smoking': smoking,
            'RiskScore': risk_score,
            'RiskLevel': risk_level
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

# Generate the data
df = generate_diabetes_data(50000)

# Save to CSV
df.to_csv('diabetes_data.csv', index=False)
print("Dataset generated and saved to 'diabetes_data.csv'")
