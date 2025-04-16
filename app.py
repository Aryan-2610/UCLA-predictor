from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
import os
app = Flask(__name__)

# Load the model from the pickle file
with open("model_final.pkl", "rb") as f:
    model = pickle.load(f)

# Home route to render the HTML form
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle prediction and show the result
@app.route("/predict", methods=['POST'])
def predict():
    # Get the input values from the form
    input_values = [float(x) for x in request.form.values()]
    
    # Convert the input values into a DataFrame with correct column names
    input_df = pd.DataFrame([input_values], columns=['GRE', 'TOEFL', 'Rating', 'SOP', 'LOR', 'CGPA', 'Research'])
    
    # Predict using the loaded model
    prediction = model.predict(input_df)[0]
    
    # Return the result back to the template
    return render_template("index.html", result=prediction)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
