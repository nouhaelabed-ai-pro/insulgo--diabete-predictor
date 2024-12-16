from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__, 
           static_url_path='', 
           static_folder='static',
           template_folder='templates')

# Load the model (we'll need to train and save it first)
try:
    model = joblib.load('model/diabetes_model.pkl')
except:
    model = None

@app.route('/')
def home():
    """Insulgo home page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Insulgo prediction endpoint"""
    if request.method == 'GET':
        return render_template('predict.html')
    
    if request.method == 'POST':
        try:
            # Get values from the form
            features = [
                float(request.form['pregnancies']),
                float(request.form['glucose']),
                float(request.form['blood_pressure']),
                float(request.form['skin_thickness']),
                float(request.form['insulin']),
                float(request.form['bmi']),
                float(request.form['diabetes_pedigree']),
                float(request.form['age'])
            ]
            
            # Make prediction
            if model:
                features_array = np.array(features).reshape(1, -1)
                prediction = model.predict(features_array)[0]
                probability = model.predict_proba(features_array)[0][1]
                
                result = {
                    'prediction': int(prediction),
                    'probability': float(probability),
                    'features': features
                }
                
                return render_template('result.html', result=result)
            else:
                return render_template('error.html', message="Model not loaded. Please train the model first.")
                
        except Exception as e:
            return render_template('error.html', message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
