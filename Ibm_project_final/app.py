# Backend - Flask API (app.py)
from flask import Flask, request, jsonify, render_template
import pickle
import re
import pandas as pd
from textblob import TextBlob
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
CORS(app)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text preprocessing function
def clean_text(text):
     # Remove special characters and make lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
    # Remove duplicate words while preserving order
    words = text.split()
    seen = set()
    deduplicated_words = [word for word in words if not (word in seen or seen.add(word))]
    return ' '.join(deduplicated_words)


def analyze_text(text):
    """Process user input, clean it, and classify using ML model."""
    cleaned_text = clean_text(text)

    # Sentiment Analysis
    sentiment = TextBlob(cleaned_text).sentiment.polarity
    sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

    # Predict classification using trained model
    text_vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vectorized)[0]

    return {
        "cleaned_text": cleaned_text,
        "sentiment": sentiment_label
    }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_text():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = analyze_text(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
