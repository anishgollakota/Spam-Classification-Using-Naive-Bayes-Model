# This program detects whether email is spam or not

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#Read csv file
df = pd.read_csv('emails.csv')
#Print 5 rows
df.head(5)

#Rows and columns
print(df.shape)

#Look into what columns you need to train
print(df.columns)

#Check for duplicates and remove them
df.drop_duplicates(inplace=True)

#Look into words that don't exist
print(df.isnull().sum())

#Downloading the nltk package for stopwords

#Removing stopwords is a key step to processing textual data and focusing on the
#word choice that actually matters for the model.
nltk.download('stopwords')

def evaluate_text(text):
  #Remove punctuation
  nopunc = [char for char in text if char not in string.punctuation]
  nopunc = ''.join(nopunc)
  
  clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
  #Return list of clean text
  return clean_words

#tokenization test
df['text'].head().apply(evaluate_text)

#Convert text collection to matrix of tokens
messages_bow = CountVectorizer(analyzer=process_text).fit_transform(df['text'])

#80% training, 20% testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(messages_bow, df['spam'], test_size=0.20, random_state=0)

#Get shape pf messages_bow
print(X_test)
messages_bow.shape

#Create and train Bayes classifier
classifier = MultinomialNB().fit(X_train, y_train)

#Print predictions
print(classifier.predict(X_train))

#Print actual values
print(y_train.values)

#Evaluate model on training set
pred = classifier.predict(X_train)
print(classification_report(y_train, pred))
print()
print('Confusion matrix: \n', confusion_matrix(y_train, pred))
print()
print('Accuracy: ', accuracy_score(y_train, pred))

#Print predictions
print(classifier.predict(X_test))

#Print actual values
print(y_test.values)

#Evaluate model on training set
prediction = classifier.predict(X_test)
print(classification_report(y_test, prediction))
print()
print('Confusion matrix: \n', confusion_matrix(y_test, prediction))
print()
print('Accuracy: ', accuracy_score(y_test, prediction))
