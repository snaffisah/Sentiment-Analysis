# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:16:43 2022

@author: snaff
"""

#a = 'I am a boy 1234567'
#print(a.replace ('1234567', ''))
# To remoce numerical data
# print(re.sub('[^a-zA-Z]', ' ', a).lower()) # alphabet is not included

#%% 
# EDA
# Step 1) Loading of data

import re
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Bidirectional
import numpy as np
import os
import datetime

#%% Path

URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'
LOG_PATH = os.path.join(os.getcwd(),'log')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')

#%%

df = pd.read_csv(URL)

review = df['review']
review_dummy = review.copy()

sentiment = df['sentiment']
sentiment_dummy = sentiment.copy()

# Step 2) Data inspection
review_dummy[3]
sentiment_dummy[3]

review_dummy[11]
sentiment_dummy[4]

# Step 3) Data Cleaning
# to remove html tags
for index, text in enumerate(review_dummy): #enumerate will return index and text to its place
    #review_dummy[index] = text.replace('<br />', '')
    review_dummy[index] = re.sub('<.*?>', '', text)
    # re.sub('<.*?>', '', review_dummy[11]) # everything inside <> will be replace    

# to convert to lowercase and split it
for index, text in enumerate(review_dummy):
    review_dummy[index] = re.sub('[^a-zA-Z]', ' ', text).lower().split()

# Step 4) Features selection
# Step 5) Data preprocessing
# Data vectorization for reviews

num_words = 10000
oov_token = '<OOV>'

# tokenizer to vectorize the words
tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(review_dummy)

# To save the tokenizer for deployment purpose
TOKENIZER_JSON_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')
token_json = tokenizer.to_json()

import json
with open(TOKENIZER_JSON_PATH,'w') as json_file:
    json.dump(token_json, json_file)

# to observe the number of words
word_index = tokenizer.word_index
print(word_index)
print(dict(list(word_index.items())[0:10]))

# to vectorize the sequences of text
review_dummy = tokenizer.texts_to_sequences(review_dummy)

pad_sequences(review_dummy, maxlen=200)
temp = [np.shape(i) for i in review_dummy] # to check the number of word inside list
np.mean(temp) # since the mean for length words 235, we choose maxlen 200

review_dummy = pad_sequences(review_dummy,
                             maxlen=200,
                             padding='post',
                             truncating='post')


# One-hot encoding for label
from sklearn.preprocessing import OneHotEncoder

one_hot_encoder = OneHotEncoder(sparse=False)
sentiment_encoded = one_hot_encoder.fit_transform(np.expand_dims(sentiment_dummy,
                                                                 axis=-1))

# train test split
from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(review_dummy, 
                                                   sentiment_encoded, 
                                                   test_size=0.3, 
                                                   random_state=123)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

one_hot_encoder.inverse_transform(np.expand_dims(y_train[0], axis=0))
# positive = [0,1]
# negative = [1,0]

#%% Model creation
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.layers import Embedding

model = Sequential()
model.add(Embedding(num_words, 64))

model.add(Bidirectional(LSTM(32,return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.summary()

#model.add(LSTM(128, input_shape=(np.shape(x_train)[1:]),
#                                 return_sequences=True))
#model.add(Dropout(0.2))
#model.add(LSTM(128))
#model.add(Dropout(0.2))
#model.add(Dense(2, activation='softmax'))
#model.summary()

# Step 3a) Callbacks
import datetime
log_dir = os.path.join(LOG_PATH, 
                       datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

from tensorflow.keras.callbacks import TensorBoard

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

#%% Compile & model fitting
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='acc')

model.fit(x_train,y_train,epochs=5,
          validation_data=(x_test,y_test), 
          callbacks=tensorboard_callback)

#%% Model Evaluation
# Append approach
#redicted = []

#for test in x_test:
#    predicted.append(model.predict(np.expand_dims(test,axis=0)))

# Pre allocation of memory approach
predicted_advanced = np.empty([len(x_test), 2])
for index, test in enumerate(x_test):
    predicted_advanced[index,:] = model.predict(np.expand_dims(test,axis=0))

#%% Model analysis
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

y_pred = np.argmax(predicted_advanced, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))
print(accuracy_score(y_true, y_pred))

#%% Model Deployment

model.save(MODEL_SAVE_PATH)