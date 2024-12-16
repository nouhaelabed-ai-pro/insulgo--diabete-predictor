from flask import Flask, render_template, request, jsonify
from flask_wtf import FlaskForm
from wtforms import FloatField, IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired, NumberRange
import joblib
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key in production

# Load the trained model, scaler, and metadata
model = joblib.load('diabetes_risk_model.joblib')
scaler = joblib.load('scaler.joblib')
metadata = joblib.load('model_metadata.joblib')
gender_encoder = joblib.load('gender_encoder.joblib')

class DiabetesForm(FlaskForm):
    age = IntegerField('Age', 
                      validators=[DataRequired(), 
                                NumberRange(min=metadata['feature_ranges']['Age']['min'],
                                          max=metadata['feature_ranges']['Age']['max'])],
                      render_kw={"placeholder": f"Enter age ({metadata['feature_ranges']['Age']['min']}-{metadata['feature_ranges']['Age']['max']})"})
    
    gender = SelectField('Gender',
                        choices=[('M', 'Male'), ('F', 'Female')],
                        validators=[DataRequired()])
    
    bmi = FloatField('BMI',
                     validators=[DataRequired(), 
                               NumberRange(min=metadata['feature_ranges']['BMI']['min'],
                                         max=metadata['feature_ranges']['BMI']['max'])],
                     render_kw={"placeholder": "Enter BMI"})
    
    fasting_glucose = FloatField('Fasting Glucose (mg/dL)',
                                validators=[DataRequired(), 
                                          NumberRange(min=metadata['feature_ranges']['FastingGlucose']['min'],
                                                    max=metadata['feature_ranges']['FastingGlucose']['max'])],
                                render_kw={"placeholder": "Enter fasting glucose"})
    
    hba1c = FloatField('HbA1c (%)',
                       validators=[DataRequired(), 
                                 NumberRange(min=metadata['feature_ranges']['HbA1c']['min'],
                                           max=metadata['feature_ranges']['HbA1c']['max'])],
                       render_kw={"placeholder": "Enter HbA1c"})
    
    blood_pressure_systolic = IntegerField('Systolic Blood Pressure',
                                         validators=[DataRequired(), 
                                                   NumberRange(min=metadata['feature_ranges']['BloodPressureSystolic']['min'],
                                                             max=metadata['feature_ranges']['BloodPressureSystolic']['max'])],
                                         render_kw={"placeholder": "Enter systolic BP"})
    
    blood_pressure_diastolic = IntegerField('Diastolic Blood Pressure',
                                          validators=[DataRequired(), 
                                                    NumberRange(min=metadata['feature_ranges']['BloodPressureDiastolic']['min'],
                                                              max=metadata['feature_ranges']['BloodPressureDiastolic']['max'])],
                                          render_kw={"placeholder": "Enter diastolic BP"})
    
    physical_activity = IntegerField('Physical Activity (days/week)',
                                   validators=[DataRequired(), 
                                             NumberRange(min=metadata['feature_ranges']['PhysicalActivity']['min'],
                                                       max=metadata['feature_ranges']['PhysicalActivity']['max'])],
                                   render_kw={"placeholder": "Days of exercise per week"})
    
    smoking = SelectField('Smoking Status',
                         choices=[('0', 'Non-smoker'), ('1', 'Smoker')],
                         validators=[DataRequired()])
    
    family_history = SelectField('Family History of Diabetes',
                               choices=[('0', 'No'), ('1', 'Yes')],
                               validators=[DataRequired()])
    
    submit = SubmitField('Predict Risk Level')

def get_risk_indicators(data):
    """Generate detailed risk indicators based on input values"""
    indicators = []
    
    # BMI Risk
    if data[2] >= 30:
        indicators.append(("BMI indicates obesity", "high"))
    elif data[2] >= 25:
        indicators.append(("BMI indicates overweight", "medium"))
    
    # Fasting Glucose Risk
    if data[3] >= 126:
        indicators.append(("Fasting glucose in diabetes range", "high"))
    elif data[3] >= 100:
        indicators.append(("Fasting glucose in prediabetes range", "medium"))
    
    # HbA1c Risk
    if data[4] >= 6.5:
        indicators.append(("HbA1c in diabetes range", "high"))
    elif data[4] >= 5.7:
        indicators.append(("HbA1c in prediabetes range", "medium"))
    
    # Blood Pressure Risk
    if data[5] >= 140 or data[6] >= 90:
        indicators.append(("Blood pressure indicates stage 2 hypertension", "high"))
    elif data[5] >= 130 or data[6] >= 80:
        indicators.append(("Blood pressure indicates stage 1 hypertension", "medium"))
    elif data[5] >= 120 and data[5] <= 129 and data[6] < 80:
        indicators.append(("Blood pressure is elevated", "medium"))
    
    # Physical Activity Risk
    if data[8] < 3:
        indicators.append(("Low physical activity level", "medium"))
    
    # Smoking Risk
    if data[9] == 1:
        indicators.append(("Current smoker", "high"))
    
    # Family History Risk
    if data[7] == 1:
        indicators.append(("Family history of diabetes", "medium"))
    
    # Age Risk
    if data[0] > 45:
        indicators.append(("Age is a significant risk factor", "medium"))
    
    return indicators

# Add new routes for the multi-page website
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    form = DiabetesForm()
    prediction_result = None
    risk_details = None
    risk_indicators = None
    
    if form.validate_on_submit():
        # Prepare the input data
        gender_encoded = gender_encoder.transform([form.gender.data])[0]
        
        input_data = np.array([
            form.age.data,
            gender_encoded,
            form.bmi.data,
            form.fasting_glucose.data,
            form.hba1c.data,
            form.blood_pressure_systolic.data,
            form.blood_pressure_diastolic.data,
            int(form.family_history.data),
            form.physical_activity.data,
            int(form.smoking.data)
        ]).reshape(1, -1)
        
        # Scale the input data
        scaled_data = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(scaled_data)[0]
        probabilities = model.predict_proba(scaled_data)[0]
        
        # Get risk indicators
        risk_indicators = get_risk_indicators(input_data[0])
        
        # Prepare risk details
        risk_details = {
            'prediction': prediction,
            'probabilities': {
                'Low': f"{probabilities[1]*100:.1f}%",
                'Medium': f"{probabilities[2]*100:.1f}%",
                'High': f"{probabilities[0]*100:.1f}%"
            }
        }
        
        prediction_result = prediction
    
    return render_template('predict.html', 
                         form=form,
                         prediction=prediction_result,
                         risk_details=risk_details,
                         risk_indicators=risk_indicators,
                         metadata=metadata['feature_ranges'])

if __name__ == '__main__':
    app.run(debug=True, port=3000, host='0.0.0.0')
