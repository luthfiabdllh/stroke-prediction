from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Load model and pipeline
model = joblib.load('model.joblib')
pipeline = joblib.load('preprocessing_pipeline.joblib')

def make_prediction(data):
    # Create DataFrame
    df = pd.DataFrame([data])

    # Preprocess
    X_prep = pipeline.transform(df)

    # Predict
    prediction = model.predict(X_prep)[0]
    probability = model.predict_proba(X_prep)[0][1]
    
    return prediction, probability

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = {
            'gender': request.form['gender'],
            'age': float(request.form['age']),
            'hypertension': int(request.form['hypertension']),
            'heart_disease': int(request.form['heart_disease']),
            'ever_married': request.form['ever_married'],
            'work_type': request.form['work_type'],
            'Residence_type': request.form['Residence_type'],
            'avg_glucose_level': float(request.form['avg_glucose_level']),
            'bmi': float(request.form['bmi']) if request.form['bmi'] else np.nan,
            'smoking_status': request.form['smoking_status']
        }

        prediction, probability = make_prediction(data)

        result = "High Risk of Stroke" if prediction == 1 else "Low Risk of Stroke"
        
        return render_template('index.html', result=result, probability=round(probability * 100, 2))

    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        # Get JSON data
        req_data = request.get_json()
        
        # Validate required fields (basic check)
        required_fields = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 
                           'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status']
        
        for field in required_fields:
            if field not in req_data:
                return jsonify({'error': f'Missing field: {field}'}), 400

        # Prepare data
        data = {
            'gender': req_data['gender'],
            'age': float(req_data['age']),
            'hypertension': int(req_data['hypertension']),
            'heart_disease': int(req_data['heart_disease']),
            'ever_married': req_data['ever_married'],
            'work_type': req_data['work_type'],
            'Residence_type': req_data['Residence_type'],
            'avg_glucose_level': float(req_data['avg_glucose_level']),
            'bmi': float(req_data['bmi']) if req_data['bmi'] is not None else np.nan,
            'smoking_status': req_data['smoking_status']
        }

        prediction, probability = make_prediction(data)
        
        return jsonify({
            'prediction': int(prediction),
            'probability': float(probability),
            'result': "High Risk of Stroke" if prediction == 1 else "Low Risk of Stroke"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
