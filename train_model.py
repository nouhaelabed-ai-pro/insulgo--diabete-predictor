<<<<<<< HEAD
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the data
print("Loading and preparing data...")
df = pd.read_csv('diabetes_data.csv')

# Prepare features and target
X = df.drop(['PatientID', 'RiskScore', 'RiskLevel'], axis=1)
y = df['RiskLevel']

# Convert categorical variables
le = LabelEncoder()
X['Gender'] = le.fit_transform(X['Gender'])
joblib.dump(le, 'gender_encoder.joblib')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
print("\nTraining Random Forest model...")
rf_model = RandomForestClassifier(n_estimators=100, 
                                max_depth=10,
                                min_samples_split=5,
                                min_samples_leaf=2,
                                random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = rf_model.predict(X_test_scaled)

# Print model performance
print("\nModel Performance Report:")
print("------------------------")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create feature importance visualization
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# Create and save feature importance plot
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance in Diabetes Risk Prediction')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')

# Save model metadata
model_metadata = {
    'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    'model_version': '2.0',
    'feature_ranges': {
        'Age': {'min': 18, 'max': 85},
        'BMI': {'min': 18.5, 'max': 45.0, 
                'ranges': {'Normal': '18.5-24.9', 'Overweight': '25.0-29.9', 'Obese': '30.0-45.0'}},
        'FastingGlucose': {'min': 70, 'max': 200, 
                          'ranges': {'Normal': '70-99', 'Prediabetes': '100-125', 'Diabetes': '≥126'}},
        'HbA1c': {'min': 4.0, 'max': 12.0,
                  'ranges': {'Normal': '<5.7', 'Prediabetes': '5.7-6.4', 'Diabetes': '≥6.5'}},
        'BloodPressureSystolic': {'min': 90, 'max': 180,
                                 'ranges': {'Normal': '90-119', 'Elevated': '120-129', 
                                          'Stage1': '130-139', 'Stage2': '≥140'}},
        'BloodPressureDiastolic': {'min': 60, 'max': 110,
                                  'ranges': {'Normal': '60-79', 'Stage1': '80-89', 'Stage2': '≥90'}},
        'PhysicalActivity': {'min': 0, 'max': 7},
        'Smoking': {'values': [0, 1]},
        'FamilyHistory': {'values': [0, 1]}
    }
}

# Save the model, scaler, and metadata
print("\nSaving model and related files...")
joblib.dump(rf_model, 'diabetes_risk_model.joblib')
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(model_metadata, 'model_metadata.joblib')

print("\nModel and related files saved successfully!")
print("Files saved: diabetes_risk_model.joblib, scaler.joblib, gender_encoder.joblib, model_metadata.joblib, feature_importance.png")
=======
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def load_data():
    """Load and preprocess the diabetes dataset"""
    try:
        # Try to load the dataset (you'll need to provide the actual data file)
        data = pd.read_csv('data/diabetes.csv')
        return data
    except FileNotFoundError:
        print("Generating synthetic data for demonstration...")
        # Generate synthetic data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'Pregnancies': np.random.randint(0, 15, n_samples),
            'Glucose': np.random.normal(120, 30, n_samples),
            'BloodPressure': np.random.normal(70, 10, n_samples),
            'SkinThickness': np.random.normal(20, 10, n_samples),
            'Insulin': np.random.normal(80, 20, n_samples),
            'BMI': np.random.normal(32, 7, n_samples),
            'DiabetesPedigreeFunction': np.random.normal(0.5, 0.3, n_samples),
            'Age': np.random.normal(35, 10, n_samples),
            'Outcome': np.random.binomial(1, 0.3, n_samples)
        }
        
        return pd.DataFrame(data)

def preprocess_data(data):
    """Preprocess the data"""
    # Separate features and target
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train):
    """Train the model"""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model"""
    score = model.score(X_test, y_test)
    print(f"Model Accuracy: {score:.2f}")
    return score

def save_model(model, scaler):
    """Save the model and scaler"""
    joblib.dump(model, 'model/diabetes_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    print("Model and scaler saved successfully!")

def main():
    """Main training pipeline"""
    print("Loading data...")
    data = load_data()
    
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    
    print("Training model...")
    model = train_model(X_train, y_train)
    
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)
    
    print("Saving model...")
    save_model(model, scaler)

if __name__ == "__main__":
    main()
>>>>>>> origin/master
