from flask import Flask, request, render_template
import pickle
import numpy as np
import re
from bs4 import BeautifulSoup

app = Flask(__name__)

# Load model and vectorizer
mnb_model = pickle.load(open('mnb.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tf.pkl', 'rb'))

# Preprocess function
def preprocess(q):
    q = str(q).lower().strip()
    q = q.replace('%', 'percent').replace('@', 'at').replace('$', 'dollar').replace('€', 'euro').replace('₹', 'rupee')
    q = re.sub(r'\W', ' ', q)
    q = BeautifulSoup(q, "html.parser").get_text()
    return q

# Helper functions
def common_words(question1, question2):
    w1 = set(map(lambda word: word.lower().strip(" "), question1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(" "), question2.split(" ")))
    return len(w1 & w2)

def total_word(question1, question2):
    w1 = set(map(lambda word: word.lower().strip(" "), question1.split(" ")))
    w2 = set(map(lambda word: word.lower().strip(" "), question2.split(" ")))
    return len(w1) + len(w2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        question1 = request.form['question1']
        question2 = request.form['question2']
        
        # Preprocess questions
        question1 = preprocess(question1)
        question2 = preprocess(question2)
        
        # TF-IDF Transformation
        q1_vec = tfidf_vectorizer.transform([question1]).toarray()
        q2_vec = tfidf_vectorizer.transform([question2]).toarray()
        
        # Calculate additional features
        que1_len = len(question1)
        que2_len = len(question2)
        que1_num_words = len(question1.split(" "))
        que2_num_words = len(question2.split(" "))
        word_total = total_word(question1, question2)
        word_common = common_words(question1, question2)
        word_share = round(word_common / word_total, 2) if word_total != 0 else 0
        
        # Reshape additional features to 2D
        additional_features = np.array([que1_len, que2_len, que1_num_words, que2_num_words,
                                        word_common, word_total, word_share]).reshape(1, -1)
        
        # Concatenate all features
        combined_vec = np.hstack((q1_vec, q2_vec, additional_features))
        
        # Predict with Naive Bayes model
        prediction = mnb_model.predict(combined_vec)
        result = "Duplicate" if prediction[0] == 1 else "Not Duplicate"
        
        return render_template('index.html', prediction_text=f'Prediction: {result}')

if __name__ == "__main__":
    app.run(debug=True)
