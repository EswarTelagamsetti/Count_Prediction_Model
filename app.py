from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

# Load the trained model and label encoder
model, label_encoder = pickle.load(open("prawn_model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age_of_pond = float(request.form['age_of_pond'])
        food_intake = float(request.form['food_intake'])
        season = request.form['season']
        
        # Encode season
        season_encoded = label_encoder.transform([season])[0]
        
        # Prepare input for prediction
        features = np.array([[age_of_pond, food_intake, season_encoded]])
        prediction = model.predict(features)[0]
        
        return jsonify({'prawn_count': round(prediction, 2)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
