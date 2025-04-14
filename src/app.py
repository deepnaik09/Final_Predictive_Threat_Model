from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import traceback
import os

app = Flask(__name__)
CORS(app)

# Load model
try:
    model_path = os.path.join(os.path.dirname(__file__), '../app/models/Predictive_model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"⚠️ Model loading error: {str(e)}")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        gender_map = {'Male': 0, 'Female': 1, 'Other': 2}
        marital_map = {'Single': 0, 'Married': 1, 'Divorced': 2}

        # Check categorical validity
        if data['gender'] not in gender_map or data['marital_status'] not in marital_map:
            return jsonify({'error': 'Invalid gender or marital status'}), 400

        # Prepare input DataFrame
        model_input = pd.DataFrame([{
            'AGE': data['age'],
            'AMT_INCOME_TOTAL': data['annual_income'],
            'AMT_CREDIT': data['loan_amount'],
            'AMT_ANNUITY': data['loan_amount'] / (data['tenure'] * 12),  # monthly
            'AMT_GOODS_PRICE': data['loan_amount'],
            'GENDER': gender_map[data['gender']],
            'MARITAL_STATUS': marital_map[data['marital_status']]
        }])

        # Reorder columns
        expected_columns = ['AGE', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
                            'AMT_GOODS_PRICE', 'GENDER', 'MARITAL_STATUS']
        model_input = model_input[expected_columns]

        # Validate input
        if model_input.isnull().any().any() or (model_input < 0).any().any():
            return jsonify({'error': 'Invalid or negative input values'}), 400

        # --- Custom threshold prediction ---
        proba = model.predict_proba(model_input.to_numpy())[0][1]
        threshold = 0.25 # Lower threshold to allow more approvals
        prediction = 1 if proba >= threshold else 0

        # Log the probability
        print(f"🧠 Predicted probability: {proba:.4f}, Prediction: {prediction}")

        return jsonify({
            'result': 'Approved' if prediction else 'Denied',
            'score': round(proba, 4)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
