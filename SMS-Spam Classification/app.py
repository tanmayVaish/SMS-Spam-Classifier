
# -- coding: utf-8 --
"""
Created on Tue Jul  7 21:53:29 2020

@author: Tanmay
"""

# Importing the libraries
from flask import Flask, render_template, url_for, request
import pandas as pd
from sklearn import feature_extraction, model_selection
from sklearn.naive_bayes import MultinomialNB
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the Model from Disc
t='transform.pkl'
m='model.pkl'
tfidf = pickle.load(open(t, 'rb'))
classifier = pickle.load(open(m, 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Reading the Data
    #data = pd.read_csv("SMSspam.csv",  encoding='latin-1')

    # Feature Extraction
    #f = feature_extraction.text.TfidfVectorizer(stop_words = 'english')
    #X = f.fit_transform(data['v2'])   # Sparse Matrix

    # Saving Model to Disc
    #pickle.dump(f, open('transform.pkl', 'wb'))

    # Test_Train Split
    #data['v1'] = data['v1'].map({'spam' : 1, 'ham' : 0})
    #X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['v1'], test_size = 0.33, random_state = 42)

    # Training Model
    #naive_bayes = MultinomialNB()
    #naive_bayes.fit(X_train, y_train)

    # Model Score
    #naive_bayes.score(X_test, y_test)

    # Saving Model to Disc
    #pickle.dump(naive_bayes, open('model.pkl', 'wb'))
    
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tfidf.transform(data).toarray()
        my_prediction = classifier.predict(vect)
    return render_template('result.html', prediction_text = my_prediction)

if __name__ == '__main__':
    app.run(debug=True)