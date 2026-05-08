from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Global variables for model and preprocessor
model = None
preprocessor = None

def create_features(df):
    """Create engineered features from raw data"""
    # Create age group feature
    df['age_group'] = pd.cut(df['age'], 
                              bins=[0, 25, 35, 45, 55, 65, 100],
                              labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'])
    
    # Create balance category feature
    df['balance_category'] = pd.cut(df['balance'],
                                      bins=[-10000, 0, 500, 2000, 5000, 100000],
                                      labels=['Negative', 'Low (0-500)', 'Medium (500-2k)', 
                                              'High (2k-5k)', 'Very High (5k+)'])
    
    # Create interaction feature
    df['campaign_previous'] = df['campaign'] * df['previous']
    
    # Create flag for multiple contacts
    df['multiple_contacts'] = (df['campaign'] > 2).astype(int)
    
    # Create flag for high balance
    df['high_balance'] = (df['balance'] > 2000).astype(int)
    
    return df

def load_artifacts():
    """Load the trained model and preprocessor"""
    global model, preprocessor
    model = joblib.load("best_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    print("Model and preprocessor loaded successfully")

@app.route('/', methods=['GET'])
def home():
    """Home endpoint - API information"""
    return jsonify({
        'service': 'Bank Term Deposit Prediction API',
        'version': '1.0',
        'status': 'running',
        'endpoints': {
            '/predict': 'POST - Send customer data to get prediction',
            '/predict_batch': 'POST - Send multiple customers',
            '/health': 'GET - Check service health'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    if model is not None and preprocessor is not None:
        return jsonify({'status': 'healthy', 'model_loaded': True})
    return jsonify({'status': 'unhealthy', 'model_loaded': False}), 503

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if a single customer will subscribe"""
    try:
        data = request.get_json()
        
        required_fields = ['age', 'job', 'marital', 'education', 'default', 
                           'balance', 'housing', 'loan', 'contact', 'day', 
                           'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        df = pd.DataFrame([data])
        
        # Create engineered features
        df = create_features(df)
        
        # Select columns in the correct order
        categorical_features = ['job', 'marital', 'education', 'default', 'housing', 
                                'loan', 'contact', 'month', 'poutcome', 'age_group', 
                                'balance_category']
        
        numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 
                              'previous', 'campaign_previous', 'multiple_contacts', 
                              'high_balance']
        
        # Keep only the columns the preprocessor expects
        df = df[categorical_features + numerical_features]
        
        X_processed = preprocessor.transform(df)
        
        prediction = model.predict(X_processed)[0]
        probability = model.predict_proba(X_processed)[0][1]
        
        response = {
            'prediction': 'SUBSCRIBE' if prediction == 1 else 'NOT SUBSCRIBE',
            'probability': round(float(probability), 4),
            'probability_percent': f"{probability*100:.1f}%",
            'recommendation': 'CALL' if probability > 0.5 else 'DO NOT CALL'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Predict for multiple customers"""
    try:
        data = request.get_json()
        
        if 'customers' not in data:
            return jsonify({'error': 'Missing customers list'}), 400
        
        customers = data['customers']
        df = pd.DataFrame(customers)
        
        # Create engineered features
        df = create_features(df)
        
        # Select columns in the correct order
        categorical_features = ['job', 'marital', 'education', 'default', 'housing', 
                                'loan', 'contact', 'month', 'poutcome', 'age_group', 
                                'balance_category']
        
        numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 
                              'previous', 'campaign_previous', 'multiple_contacts', 
                              'high_balance']
        
        # Keep only the columns the preprocessor expects
        df = df[categorical_features + numerical_features]
        
        X_processed = preprocessor.transform(df)
        predictions = model.predict(X_processed)
        probabilities = model.predict_proba(X_processed)[:, 1]
        
        results = []
        for i in range(len(customers)):
            results.append({
                'customer_id': customers[i].get('customer_id', i),
                'prediction': 'SUBSCRIBE' if predictions[i] == 1 else 'NOT SUBSCRIBE',
                'probability': round(float(probabilities[i]), 4),
                'recommendation': 'CALL' if probabilities[i] > 0.5 else 'DO NOT CALL'
            })
        
        return jsonify({'results': results, 'total': len(results)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Load artifacts when starting
load_artifacts()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
