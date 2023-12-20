from flask import Flask, jsonify, request
from pathlib import Path
import joblib
import pandas as pd

import sklearn 

app = Flask(__name__)

# Load the RandomForestClassifier model
model_path = Path(__file__).parent / 'models' / 'RFC_model_0001.joblib'
model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        json_data = request.get_json()

        # Prepare the data for prediction
        input_data = pd.json_normalize(json_data)  # Assuming the JSON is in a flat format

        # Make predictions
        predictions = model.predict(input_data)

        # Convert predictions to a list (or any format you prefer)
        result = predictions.tolist()

        return jsonify({'predictions': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
