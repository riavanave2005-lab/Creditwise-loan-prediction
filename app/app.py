from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import sys
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

app = Flask(__name__)

# Load model
model_path = os.path.join(PROJECT_ROOT, 'models', 'best_model.pkl')
model = None

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        print(f"✅ Model loaded successfully from: {model_path}")
        print(f"   Model type: {type(model)}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ Model file not found at: {model_path}")

# Load preprocessor and feature engineer
preprocessor_path = os.path.join(PROJECT_ROOT, 'models', 'preprocessor.pkl')
engineer_path = os.path.join(PROJECT_ROOT, 'models', 'feature_engineer.pkl')

try:
    preprocessor = joblib.load(preprocessor_path) if os.path.exists(preprocessor_path) else None
    print(f"✅ Preprocessor loaded: {preprocessor is not None}")
except:
    preprocessor = None
    print("⚠️ Preprocessor not loaded")

try:
    engineer = joblib.load(engineer_path) if os.path.exists(engineer_path) else None
    print(f"✅ Feature engineer loaded: {engineer is not None}")
except:
    engineer = None
    print("⚠️ Feature engineer not loaded")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("\n" + "="*50)
        print("🔮 New Prediction Request")
        print("="*50)
        
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.',
                'prediction': 0,
                'probability': 0.0,
                'status': 'Error'
            }), 500
        
        # Get form data
        form_data = request.form
        print(f"📋 Form data: {dict(form_data)}")
        
        # Extract values
        applicant_income = float(form_data.get('applicant_income', 5000))
        coapplicant_income = float(form_data.get('coapplicant_income', 2000))
        loan_amount = float(form_data.get('loan_amount', 150))
        loan_term = float(form_data.get('loan_term', 360))
        credit_history = int(form_data.get('credit_history', 1))
        gender = form_data.get('gender', 'Male')
        married = form_data.get('married', 'Yes')
        dependents = form_data.get('dependents', '0')
        education = form_data.get('education', 'Graduate')
        self_employed = form_data.get('self_employed', 'No')
        property_area = form_data.get('property_area', 'Urban')
        
        # Create DataFrame
        data = {
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_term],
            'Credit_History': [credit_history],
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'Property_Area': [property_area]
        }
        
        input_df = pd.DataFrame(data)
        print(f"📊 Input DataFrame: {input_df.to_dict()}")
        
        # Apply feature engineering if available
        if engineer:
            try:
                input_df = engineer.engineer_features(input_df)
                print(f"🔧 After feature engineering: {input_df.shape}")
            except Exception as e:
                print(f"⚠️ Feature engineering error: {e}")
        
        # Apply encoding if available
        if preprocessor:
            try:
                input_df = preprocessor.encode_categorical(input_df)
                print(f"🔢 After encoding: {input_df.shape}")
            except Exception as e:
                print(f"⚠️ Encoding error: {e}")
        
        # Ensure all features are numeric
        for col in input_df.columns:
            if input_df[col].dtype == 'object':
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
        
        # Handle missing values
        input_df = input_df.fillna(0)
        input_df = input_df.replace([np.inf, -np.inf], 0)
        
        print(f"📊 Final features: {input_df.columns.tolist()}")
        print(f"📊 Feature count: {input_df.shape[1]}")
        
        # Check if we have the right number of features
        # If not, pad with zeros
        if input_df.shape[1] < 19:
            print(f"⚠️ Only {input_df.shape[1]} features, padding to 19")
            for i in range(19 - input_df.shape[1]):
                input_df[f'feature_{i}'] = 0
        
        # Take first 19 features if we have more
        if input_df.shape[1] > 19:
            print(f"⚠️ Too many features ({input_df.shape[1]}), taking first 19")
            input_df = input_df.iloc[:, :19]
        
        # Convert to numpy array
        input_array = input_df.values.astype(np.float32)
        print(f"📊 Input array shape: {input_array.shape}")
        
        # Make prediction
        prediction = int(model.predict(input_array)[0])
        probability = float(model.predict_proba(input_array)[0][1])
        
        print(f"✅ Prediction: {prediction}")
        print(f"✅ Probability: {probability:.4f}")
        print("="*50)
        
        return jsonify({
            'prediction': prediction,
            'probability': probability,
            'status': 'Approved' if prediction == 1 else 'Rejected'
        })
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'prediction': 0,
            'probability': 0.0,
            'status': 'Error'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
