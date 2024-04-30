import streamlit as st
import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('chatbot-dataset/intent.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "chatbot-dataset/data.pth"
data = torch.load(FILE)



def display_chat_history():
    """Displays the chat history"""
    for message in st.session_state['chat_history']:
        if message["author"] == "user":
            messages.chat_message("user").write(messages['content'])
        else:
            messages.chat_message("Assist").write(messages['content'])


history = st.session_state['chat_history'] = []

def update_chat_history(user_input, response):
  """Updates the chat history with new messages"""

  history.append({"author": "user", "content": user_input})
  
  history.append({"author": "assistant", "content": response})

  
  st.write(display_chat_history)




input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
sentence = ''
bot_name = "Sam"



print("Let's chat! (type 'quit' to exit)")


message = st.chat_input("Type here..")



if message:
    
    sentence = tokenize(message)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)


    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["intent"]:
                messages = st.container(height=300)
                #chatResponses.append(f"{bot_name}: {random.choice(intent['responses'])}")
                messages.chat_message("user").write(message)
                messages.chat_message("assistant").write(f"{random.choice(intent['responses'])}")
                #st.chat_message(f"{bot_name}: {random.choice(intent['responses'])}")
