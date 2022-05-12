# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:44:23 2022

This train.py python file trains the sentiment to determine if the review
is positive on negative. 

@author: snaff
"""

import pandas as pd
from sentiment_analysis_modules import (ExploratoryDataAnalysis, ModelCreation, 
                                        ModelEvaluation)
from sklearn.preprocessing import OneHotEncoder
import os
import numpy as np
from sklearn.model_selection import train_test_split
import datetime
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'
TOKEN_SAVE_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')
LOG_PATH = os.path.join(os.getcwd(),'log')
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
 
# EDA
# Step 1) Import data

df = pd.read_csv(URL)
review = df['review']
sentiment = df['sentiment']

# Step 2) Data Cleaning

eda = ExploratoryDataAnalysis()
review = eda.remove_tags(review) # Remove tags
review = eda.lower_split(review) # Convert to lower case & split

# Step 3) Features Selection
# Step 4) Data vectorization

review = eda.sentiment_tokenizer(review, TOKEN_SAVE_PATH)
review = eda.sentiment_pad_sequence(review)

# Step 5) Preprocessing
#One hot encoder
one_hot_encoder = OneHotEncoder(sparse=False)
sentiment = one_hot_encoder.fit_transform(np.expand_dims(sentiment,axis=-1))

len(np.unique(sentiment))

# Train Test Split
# x = review, y = sentiment

x_train, x_test, y_train, y_test = train_test_split(review, sentiment,
                                                    test_size = 0.3,
                                                    random_state = 123)

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# from here you will know that [0,1] is positive, [1,0] is negative
print(y_train[0])
print(one_hot_encoder.inverse_transform(np.expand_dims(y_train[0], axis=0)))

#%% model creation

mc = ModelCreation()
num_words = 10000

model = mc.lstm_layer(num_words, nb_categories)
log_dir = os.path.join(LOG_PATH, 
                       datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
#%% Compile & model fitting
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='acc')

model.fit(x_train,y_train,epochs=5,
          validation_data=(x_test,y_test), 
          callbacks=tensorboard_callback)

#%% Model Evaluation
# Pre allocation of memory approach
predicted_advanced = np.empty([len(x_test), 2])
for index, test in enumerate(x_test):
    predicted_advanced[index,:] = model.predict(np.expand_dims(test,axis=0))

#%% Model analysis
y_pred = np.argmax(predicted_advanced, axis=1)
y_true = np.argmax(y_test, axis=1)

me = ModelEvaluation()
me.report_metrics(y_true, y_pred)

#%% Model Deployment
model.save(MODEL_SAVE_PATH)












