**🧠 8th Standard History Chatbot**
A smart AI-powered educational chatbot designed specifically for 8th-grade history students. This project utilizes OCR and NLP techniques to extract questions and answers from textbook content, enabling students to interact with their curriculum in a conversational and engaging way.
**

📚 Features
**

🤖 AI Chatbot: Responds to history-related questions from the first two lessons of the 8th-grade textbook.

🔢 OCR Integration: Converts textbook pages (PDF/images) to text using Tesseract OCR.

🤮 NLP Pipeline: Preprocessing includes stop word removal, stemming, lemmatization, and TF-IDF-based matching.

⚡ Keyword Detection: Direct keyword support for fast access to facts (e.g., "1857", "Sir William Jones").

📂 Structured Dataset: All Q&A pairs are stored in a CSV format for easy maintenance and scalability.

🎓 Student-Centric: Designed to enhance textbook comprehension in an interactive format.


**🚀 How It Works

OCR Extraction**

Textbook PDFs/images are scanned using Tesseract OCR to extract raw text.

**Text Processing**
Cleaned and split into potential questions and answers using NLP tools like NLTK.

**Dataset Creation**
All valid Q&A pairs are saved into a CSV file (history_dataset.csv).

**Chatbot Interaction**
User questions are matched using TF-IDF and cosine similarity.
Best matching answer is retrieved and displayed.

**🧰 Tech Stack**

Language: Python 3.x
Libraries: pandas, sklearn, nltk, numpy
OCR: Tesseract OCR (can be swapped with Google Vision API)
NLP: TF-IDF Vectorizer, Lemmatizer, Stemmer, Stopword filtering

🎯 Example Keywords Supported

1857 → "Revolt of 1857"
Sir William Jones → "Founder of Asiatic Society of Bengal"
Powada, Cellular Jail, Material sources etc.

**🔧 Setup Instructions**

Install Dependencies
pip install pandas scikit-learn nltk numpy pytesseract

Download NLTK Resources

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

Run the Chatbot
python chatbot.py

🚫 Limitations
Answers are limited to content in Lessons 1 & 2 of the textbook.

OCR accuracy may vary depending on the quality of the scan.

🚀 Future Enhancements
Expand dataset to cover full textbook.
Add support for video explanations and interactive quizzes.
Implement voice-based input/output.

📄 License
MIT License

🙏 Acknowledgements
Inspired by the Maharashtra State Board 8th Standard History Curriculum.
Powered by Open Source technologies and tools.

Happy Learning! 🎓

