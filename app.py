from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from flask_cors import CORS
from difflib import get_close_matches


with open('random_forest_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

with open('voting_classifier_model.pkl', 'rb') as model_file:
    voting_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

data = pd.read_csv('preprocessed.csv')
data['name'] = data['name'].str.strip().str.lower()
data['label'] = data['label'].str.strip().str.lower()

app = Flask(__name__)
CORS(app) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.json.get('input_text', '').strip().lower()
    column_to_predict = request.json.get('column_to_predict', 'label')

    if not input_text:
        return jsonify({'suggestions': []})

    similar_names = get_close_matches(input_text, data['name'].dropna().str.lower().unique(), n=5, cutoff=0.5)

    similar_matches = data[data['name'].str.lower().isin(similar_names)]
    if not similar_matches.empty:
        suggestions = [
            f"{row['name']} ({row['label']})" if pd.notna(row['label']) else row['name']
            for _, row in similar_matches.iterrows()
        ]
        return jsonify({'suggestions': suggestions})

    if column_to_predict == 'label':
        matches = data[data['name'].str.contains(input_text, na=False)]
    else:
        matches = data[data['label'].str.contains(input_text, na=False)]

    suggestions = [
        f"{row['name']} ({row['label']})" if pd.notna(row['label']) else row['name']
        for _, row in matches.iterrows()
    ]

    return jsonify({'suggestions': suggestions})


if __name__ == '__main__':
    app.run(debug=True) 