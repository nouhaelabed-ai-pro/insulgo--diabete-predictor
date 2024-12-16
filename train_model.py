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
