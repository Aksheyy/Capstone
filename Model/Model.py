import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Step 1: Load the dataset
file_path = "history_dataset.csv"  # Update with your file's path
data = pd.read_csv(file_path)

# Ensure data is in expected format
assert "Question" in data.columns and "Answer" in data.columns, "CSV must have 'Question' and 'Answer' columns."

# Step 2: Preprocess the data
questions = data['Question'].tolist()
answers = data['Answer'].tolist()

# Step 3: Train the model (TF-IDF Vectorizer)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(questions)

# Helper function to suggest similar questions
def suggest_similar_questions(query, top_n=3):
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    similar_indices = similarities.argsort()[0, -top_n:][::-1]

    suggestions = []
    for idx in similar_indices:
        if similarities[0][idx] > 0:  # Only consider positive similarity scores
            suggestions.append(questions[idx])
    return suggestions

# Step 4: Define the chatbot function
def chatbot():
    print("Welcome to the History Chatbot!")
    print("You can ask questions from the textbook, and I'll try my best to answer them.")
    print("If you're unsure how to phrase your question, I can suggest similar ones.")
    print("Type 'exit' to end the session.")

    while True:
        user_query = input("\nType your question: ").strip()
        if user_query.lower() == "exit":
            print("Thank you for using the History Chatbot. Goodbye!")
            break

        # Process the user's query
        query_vector = vectorizer.transform([user_query])
        similarities = cosine_similarity(query_vector, tfidf_matrix)

        # Get the highest similarity score
        max_similarity_index = np.argmax(similarities)
        max_similarity_score = similarities[0][max_similarity_index]

        # Threshold to determine if the question is out of scope
        threshold = 0.3  # You can adjust this value

        if max_similarity_score > threshold:
            response = answers[max_similarity_index]
            print(f"Answer: {response}")
        else:
            print("I'm not sure about that. It seems your question isn't directly from the textbook.")
            print("Here are some similar questions you could try:")
            suggestions = suggest_similar_questions(user_query)
            for i, suggestion in enumerate(suggestions, start=1):
                print(f"{i}. {suggestion}")
            print("Please refine your question based on these suggestions or ask something else.")

# Step 5: Run the chatbot
if __name__ == "__main__":
    chatbot()
