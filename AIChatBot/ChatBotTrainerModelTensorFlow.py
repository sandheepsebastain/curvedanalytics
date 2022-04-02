# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 19:50:04 2022

@author: sebasa
"""
#importing the libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import json
import nltk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM , Dense,GlobalMaxPooling1D,Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import pickle



#importing the chat dataset
#Storing the questions/inputs and the corresponding tags into the data dataframe
with open('content.json') as content:
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
#converting to dataframe
data = pd.DataFrame({"inputs":inputs,
                     "tags":tags})
print(data)

###########Inputs transformation#######################
#Cleaning up the inputs to make it ready for training
import string
#Converting all to lower case and removing punctuations if they exist
data['inputs'] = data['inputs'].apply(lambda wrd:[ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])

#putting in back to gether
data['inputs'] = data['inputs'].apply(lambda wrd: ''.join(wrd))

#tokenize the data
from tensorflow.keras.preprocessing.text import Tokenizer
#Tokenizer num words tell the text_to_sequence function how many of the top num_words you want to convert
#If num_words is 1 the text_to_sequence will only return a vectorized sequence with the top 1 word
#fit_on_text will provide a word index. Giving an id for each word in your corpus. In this case the chat json
#texts_to_sequence will use this word index to convert all our sentences to a number vector
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['inputs'])
train = tokenizer.texts_to_sequences(data['inputs'])

#apply padding
#Padding makes sure all the sentence vectors are of the same size
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(train)

#######################Tags Transformation#############
#encoding the outputs
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

#input length
input_shape = x_train.shape[1]
print(input_shape)
#define vocabulary
vocabulary = len(tokenizer.word_index)
print("number of unique words : ",vocabulary)
#output length
output_length = le.classes_.shape[0]
print("output length: ",output_length)


#creating the model
i = Input(shape=(input_shape,))
x = Embedding(vocabulary+1,10)(i)
x = LSTM(10,return_sequences=True)(x)
x = Flatten()(x)
x = Dense(output_length,activation="softmax")(x)
model  = Model(i,x)
#compiling the model
model.compile(loss="sparse_categorical_crossentropy",optimizer='adam',metrics=['accuracy'])
#training the model
train = model.fit(x_train,y_train,epochs=200)

# saving toeknizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('labelencoder.pickle', 'wb') as handle:
    pickle.dump(le, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#saving model
model.save('chatbottrainedmodel')
