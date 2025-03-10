from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and label encoder
model, label_encoder = pickle.load(open("prawn_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        age = float(data['Age_of_Pond'])
        food = float(data['Food_Intake'])
        season = data['Season']

        # Convert season to encoded number
        season_encoded = label_encoder.transform([season])[0]

        # Ensure correct feature names to match training
        features = pd.DataFrame([[age, food, season_encoded]], columns=['Age_of_Pond', 'Food_Intake', 'Season'])

        prediction = model.predict(features)[0]
        return jsonify({'prediction': round(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)
