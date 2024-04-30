import random
import pickle

import numpy as np
import pandas as pd 
from nltk.tokenize import RegexpTokenizer

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop


model = load_model("text_generator.h5")

text_df = pd.read_csv("fake_or_real_news.csv")

text = list(text_df.text.values)
joined_text = " ".join(text)

partial_text = joined_text[:100000]

tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(partial_text.lower())

unique_tokens = np.unique(tokens)
unique_token_index = {token: index for index, token in enumerate(unique_tokens)}

n_words = 10
input_words = []
next_words = []

for i in range(len(tokens) - n_words):
    input_words.append(tokens[i:i + n_words])
    next_words.append(tokens[i + n_words]) 

def predict_next_word(input_text, n_best):
    input_text = input_text.lower()
    X = np.zeros((1,n_words, len(unique_tokens)))
    for i, word in enumerate(input_text.split()):
        X[0, i, unique_token_index[word]] = 1

    predictions = model.predict(X)[0]
    return np.argpartition(predictions, n_best)[-n_best:]

user_input = input(" Your sentence :")

print(f" Generator : {predict_next_word(user_input,5)}")