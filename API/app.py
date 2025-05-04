from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# Download required resources
nltk.download('stopwords')
nltk.download('wordnet')

# NLP setup
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load and process dataset
file_path = "History_Dataset.csv"
data = pd.read_csv(file_path)

assert "Question" in data.columns and "Answer" in data.columns, "CSV must have 'Question' and 'Answer' columns."

def preprocess(text):
    tokens = [word for word in text.split() if word.lower() not in stop_words]
    stemmed = [stemmer.stem(word) for word in tokens]
    lemmatized = [lemmatizer.lemmatize(word) for word in stemmed]
    return ' '.join(lemmatized)

data['ProcessedQuestion'] = data['Question'].apply(preprocess)
questions = data['ProcessedQuestion'].tolist()
answers = data['Answer'].tolist()

# TF-IDF vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

# Chatbot endpoint
@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get("question", "")
    if not user_input:
        return jsonify({"error": "Question is required."}), 400

    query = preprocess(user_input)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)

    max_index = np.argmax(similarities)
    max_score = similarities[0][max_index]

    threshold = 0.7

    if max_score > threshold:
        response = answers[max_index]
    else:
        response = "I'm not sure about that. Try using keywords or rephrasing your question."

    return jsonify({"answer": response})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
