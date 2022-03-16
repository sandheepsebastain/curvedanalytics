# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 09:00:54 2022

@author: sebasa
"""
import random
import string
import pickle
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import os
#encoding the outputs
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

#apply padding
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open(os.getcwd()+'/AIChatBot/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
with open(os.getcwd()+'/AIChatBot/labelencoder.pickle', 'rb') as handle:
    le = pickle.load(handle)
    
model = load_model(os.getcwd()+"/AIChatBot/chatbottrainedmodel")

#importing the dataset
with open(os.getcwd()+'\\AIChatBot\\content.json') as content:
  data1 = json.load(content)
#getting all the data to lists
tags = []
inputs = []
responses={}
for intent in data1['intents']:
  responses[intent['tag']]=intent['responses']
  for lines in intent['input']:
    inputs.append(lines)
    tags.append(intent['tag'])
    
def getResponse(szMessage):
    texts_p = []
    prediction_input=szMessage
    prediction_input = [letters.lower() for letters in prediction_input if letters not in string.punctuation]
    prediction_input = ''.join(prediction_input)
    texts_p.append(prediction_input)
    #tokenizing and padding
    prediction_input = tokenizer.texts_to_sequences(texts_p)
    prediction_input = np.array(prediction_input).reshape(-1)
    prediction_input = pad_sequences([prediction_input],8)
    
    output = model.predict(prediction_input)
    output = output.argmax()
    #finding the right tag and predicting
    response_tag = le.inverse_transform([output])[0]
    
    return random.choice(responses[response_tag])
