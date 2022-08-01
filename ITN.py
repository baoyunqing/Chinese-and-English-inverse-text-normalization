#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import gc
from nltk import FreqDist
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Embedding, Bidirectional
from tensorflow.keras.optimizers import RMSprop
from keras.layers.recurrent import LSTM
from utils import *
import pandas as pd

input_vocab_size = 250
target_vocab_size = 110    # 1000 in full dataset
# target_vocab_size = 1000
num_samples = 500000
context_size = 3
padding_entity = [0]
self_sil_retention_percent = 0.5
X_seq_len = 60
y_seq_len = 20
hidden = 256
layers = 2
epochs = 5   # used just 1 for full dataset
batch_size = 8
val_split = 0.1
learning_rate = 0.1

model = Sequential()

# creating encoder network
model.add(Embedding(input_vocab_size+2, hidden, input_length=X_seq_len, mask_zero=True))
print('Embedding layer created')
model.add(Bidirectional(LSTM(hidden, return_sequences = True), merge_mode = 'concat'))
model.add(Bidirectional(LSTM(hidden, return_sequences = True), merge_mode = 'concat'))
model.add(Bidirectional(LSTM(hidden), merge_mode = 'concat'))
model.add(RepeatVector(y_seq_len))
print('Encoder layer created')

# creating decoder network
for _ in range(layers):
    model.add(LSTM(hidden, return_sequences=True))
model.add(TimeDistributed(Dense(target_vocab_size+1)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
print('Decoder layer created')

# load training data
X_train_data = pd.read_csv("ITN.csv")

X_train_data['before'] = X_train_data['before'].apply(str)
X_train_data['after'] = X_train_data['after'].apply(str)

X_train_data = X_train_data.iloc[:num_samples]


# create vocabularies
# target vocab
y = list(np.where(X_train_data['class'] == "PUNCT", "sil.",
      np.where(X_train_data['before'] == X_train_data['after'], "<self>",
               X_train_data['after'])))

y = [token.split() for token in y]
#y = [list(token) for token in y]

dist = FreqDist(np.hstack(y))
temp = dist.most_common(target_vocab_size-1)
temp = [word[0] for word in temp]
temp.insert(0, 'ZERO')
temp.append('UNK')

target_vocab = {word:ix for ix, word in enumerate(temp)}
target_vocab_reversed = {ix:word for word,ix in target_vocab.items()}
print(len(target_vocab))

# input vocab
X = list(X_train_data['before'])
X = [token.split() for token in X]

dist = FreqDist(np.hstack(X))
temp = dist.most_common(input_vocab_size-1)
temp = [char[0] for char in temp]
temp.insert(0, 'ZERO')
temp.append('<norm>')
temp.append('UNK')

input_vocab = {char:ix for ix, char in enumerate(temp)}
print(len(input_vocab))

gc.collect()

X = index(X, input_vocab)
y = index(y, target_vocab)

# adding a context window of 3 words in input, with token separated by <norm>
X = add_context_window(X, context_size, padding_entity, input_vocab)

# padding
X = padding_batchwise(X, X_seq_len)
y = padding_batchwise(y, y_seq_len)

# convert to integer array, batch-wise (converting full data to array at once takes a lot of time)
X = np.array(X)
y = np.array(y)
y_sequences = np.asarray(sequences(y, y_seq_len, target_vocab))

print(X.shape, y_sequences.shape)

from keras.callbacks import ModelCheckpoint, EarlyStopping
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# fitting the model on the validation data with batch size set to 128 for a total of 5 epochs:
print('Fitting model...')
checkpointer = ModelCheckpoint(filepath='saved_model.hdf5', verbose=1, save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=1, mode='auto')
callbacks_list = [checkpointer, earlystop]

history = model.fit(X, y_sequences, batch_size=batch_size, epochs=epochs, validation_split=val_split, callbacks=callbacks_list, verbose=1)

