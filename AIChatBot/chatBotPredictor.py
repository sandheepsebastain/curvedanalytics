# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 18:30:42 2022

@author: sebasa
"""
import pickle
import re
import torch
import os
import pdb


from AIChatBot.BERT_Arch import BERT_Arch
from transformers import DistilBertTokenizer, DistilBertModel
import numpy as np
import json
import random
import pdb





bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
model = BERT_Arch(bert)
with open(os.getcwd()+'\AIChatBot\intents.json', 'r') as f:
  data = json.load(f)
  
with open(os.getcwd()+'\AIChatBot\chatBotAIModel.p', 'rb') as f:
    dictModel = pickle.load(f)
    
tokenizer=dictModel['tokenizer']
model.load_state_dict(torch.load(os.getcwd()+'\AIChatBot\model.pt'))
model.eval()
le=dictModel['labelEncoder']
max_seq_len = 14




def get_prediction(str):
    
    str = re.sub(r'[^a-zA-Z ]+', '', str)
    test_text = [str]
    
    tokens_test_data = tokenizer(
    test_text,
    max_length = max_seq_len,
    pad_to_max_length=True,
    truncation=True,
    return_token_type_ids=False
    )
    test_seq = torch.tensor(tokens_test_data['input_ids'])
    test_mask = torch.tensor(tokens_test_data['attention_mask'])
    
    preds = None
    with torch.no_grad():
      preds = model(test_seq, test_mask)
    preds = preds.detach().cpu().numpy()
    preds = np.argmax(preds, axis = 1)
    print('Intent Identified: ', le.inverse_transform(preds)[0])
    return le.inverse_transform(preds)[0]


def get_response(message): 
  intent = get_prediction(message)
  for i in data['intents']: 
    if i["tag"] == intent:
      result = random.choice(i["responses"])
      break
  print(f"Response : {result}")
  return "Intent: "+ intent + '\n' + "Response: " + result

