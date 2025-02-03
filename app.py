import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # Get user input and convert to integers
    int_features = [int(x) for x in request.form.values()]

    # Define feature names to match the model's training data
    feature_names = ['experience', 'test_score', 'interview_score']

    # Convert input into a Pandas DataFrame with correct column names
    final_features = pd.DataFrame([int_features], columns=feature_names)

    # Make prediction
    prediction = model.predict(final_features)

    # Format output
    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text=f'Employee Salary should be $ {output}')



if __name__ == "__main__":
    app.run(debug=True)