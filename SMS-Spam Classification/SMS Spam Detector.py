# -- coding: utf-8 --
"""
Created on Thu Jul  2 21:33:21 2020

@author: LEON
"""

# Importing the libraries
import pandas as pd
from sklearn import feature_extraction, model_selection
from sklearn.naive_bayes import MultinomialNB
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Reading the Data
data = pd.read_csv("SMSspam.csv",  encoding='latin-1')

# Feature Extraction
f = feature_extraction.text.TfidfVectorizer(stop_words = 'english')
X = f.fit_transform(data['v2'])   # Sparse Matrix

# Saving Model to Disc
pickle.dump(f, open('transform.pkl', 'wb'))

# Test_Train Split
data['v1'] = data['v1'].map({'spam' : 1, 'ham' : 0})
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, data['v1'], test_size = 0.33, random_state = 42)

# Training Model
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)

# Model Score
naive_bayes.score(X_test, y_test)

# Saving Model to Disc
pickle.dump(naive_bayes, open('model.pkl', 'wb'))