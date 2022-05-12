# -*- coding: utf-8 -*-
"""
Created on Wed May 11 15:07:08 2022

@author: snaff
"""

from tensorflow.keras.models import load_model
import os
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import re

MODEL_PATH = os.path.join(os.getcwd(), 'model.h5')

sentiment_classifier = load_model(MODEL_PATH)
sentiment_classifier.summary()

#%% Tokenizer loading
JSON_PATH = os.path.join(os.getcwd(), 'tokenizer_data.json')
with open(JSON_PATH, 'r') as json_file:
    loaded_tokenizer = json.load(json_file)

#%% Deploy

new_review = ['<br \>I received the parcel a bit late, however...item was in\
    good condition and work perfectly. Easy to use and can charge it multiple\
        time. I would suggest this with my friends and family. It worth it.\
            Im satisfied.<br \>']     
            
# Data Cleaning
# to remove html tags
for index, text in enumerate(new_review): #enumerate will return index and text to its place
    new_review[index] = re.sub('<.*?>', '', text)

# to convert to lowercase and split it
for index, text in enumerate(new_review):
    new_review[index] = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    
# To vectorize the new review
loaded_tokenizer = tokenizer_from_json(loaded_tokenizer)

# to vectorize the review into integers
new_review = loaded_tokenizer.texts_to_sequences(new_review)

# to pad the data to ensure every row of data has equal length
new_review = pad_sequences(new_review,
                           maxlen=200,
                           truncating='post',
                           padding='post')

#%% model prediction
outcome = sentiment_classifier.predict(np.expand_dims(new_review, axis=-1))
print(outcome)
print(np.argmax(outcome))

sentiment_dict = {1:'positive', 0:'negative'}
print('This review is ' + sentiment_dict[np.argmax(outcome)])
# positive = [0,1]
# negative = [[]]
