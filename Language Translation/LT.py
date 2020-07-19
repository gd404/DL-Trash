import numpy as np
from keras.models import Model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, LSTM, Input, TimeDistributed, Activation, Bidirectional, Dropout
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras.losses import sparse_categorical_crossentropy
from string import punctuation
import re
from collections import Counter

with open('data/small_vocab_en', 'r') as f:
    file_english = f.readlines()
with open('data/small_vocab_fr', 'r') as g:
    file_french = g.readlines()

# to remove the first b'' and last .\n
english_sentences = [str(line)[:-2] for line in file_english]
french_sentences = [str(line)[:-2] for line in file_french]  

english_counter = Counter([word for sentence in english_sentences for word in sentence.split()])
french_counter = Counter([word for sentence in french_sentences for word in sentence.split()])

print('{} English words.'.format(len([word for sentence in english_sentences for word in sentence.split()])))
print('{} unique English words.'.format(len(english_counter)))
print('10 Most common words in the English dataset:')
print('"' + '" "'.join(list(zip(*english_counter.most_common(10)))[0]) + '"')

print('{} French words.'.format(len([word for sentence in french_sentences for word in sentence.split()])))
print('{} unique French words.'.format(len(french_counter)))
print('10 Most common words in the French dataset:')
print('"' + '" "'.join(list(zip(*french_counter.most_common(10)))[0]) + '"')

def tokenize(x):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(x)
    return tokenizer.texts_to_sequences(x), tokenizer

# Tokenize Example output
text_sentences = [
    'Colors in a rainbow are  collectively called as VIBGYOR .',
    'for this purpose, the report has been modified .',
    'I ate all the apples .']
text_tokenized, text_tokenizer = tokenize(text_sentences)
print(text_tokenizer.word_index)


for sample_i, (sent, token_sent) in enumerate(zip(text_sentences, text_tokenized)):
    print('Sequence {} in x'.format(sample_i + 1))
    print('  Input:  {}'.format(sent))
    print('  Output: {}'.format(token_sent))

def pad(x, length=None):
    return pad_sequences(x, maxlen = length, padding='post')

def preprocess(x, y):
    preprocess_x, x_token = tokenize(x)
    preprocess_y, y_token = tokenize(y)
    
    preprocess_x = pad(preprocess_x)
    preprocess_y = pad(preprocess_y)
    
    # Keras's sparse_categorical_crossentropy function requires the labels to be in 3 dimensions
    preprocess_y = preprocess_y.reshape(*preprocess_y.shape, 1)
    
    
    return preprocess_x, preprocess_y, x_token, y_token

preproc_english_sentences, preproc_french_sentences, english_tokenizer, french_tokenizer = preprocess(english_sentences, french_sentences)
    
max_english_sequence_length = preproc_english_sentences.shape[1]
max_french_sequence_length = preproc_french_sentences.shape[1]
english_vocab_size = len(english_tokenizer.word_index)
french_vocab_size = len(french_tokenizer.word_index)

print('Data Preprocessed')
print("Max English sentence length:", max_english_sequence_length)
print("Max French sentence length:", max_french_sequence_length)
print("English vocabulary size:", english_vocab_size)
print("French vocabulary size:", french_vocab_size)


def logits_to_text(logits, tokenizer):
    index_to_words = {i:word for word, i in tokenizer.word_index.items()}
    index_to_words[0] = '<PAD>'
    return ' '.join([index_to_words[prediction] for prediction in np.argmax(logits, 1)])

#Simple Model
def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    learning_rate=0.005
    model = Sequential()
    model.add(LSTM(256, input_shape = input_shape[1:], return_sequences=True))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Dense(french_vocab_size+1, activation='softmax')))
    
    model.compile(optimizer=Adam(learning_rate), loss = sparse_categorical_crossentropy, metrics=['accuracy'])
    return model

tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

model = simple_model(tmp_x.shape, max_french_sequence_length, english_vocab_size, french_vocab_size)

model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=10, validation_split=0.2)

res = model.predict(tmp_x[:1])[0]

ans = logits_to_text(res, french_tokenizer)

print('Predicted Translation', logits_to_text(model.predict(tmp_x[:1])[0], french_tokenizer))

print('Actual translation', french_sentences[0])

#Embeding Model
def embedding_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    learning_rate = 0.005
    model = Sequential()
    model.add(Embedding(english_vocab_size+1, 256, input_length = input_shape[1]))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Dense(french_vocab_size+1, activation='softmax')))
    
    model.compile(optimizer=Adam(learning_rate), loss = sparse_categorical_crossentropy, metrics=['accuracy'])
    return model

tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape(-1, preproc_french_sentences.shape[-2])

model = embedding_model(tmp_x.shape, preproc_french_sentences.shape[1], english_vocab_size, french_vocab_size)
model.summary()
print('model input shape is:-- ', model.input_shape)
print('model output shape is:-- ', model.output_shape)
model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs=20, validation_split=0.25)


#trying bidirectional lstm
def bidirectional_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    learning_rate = 0.005
    model = Sequential()
    model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape = input_shape[1:]))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(Dropout(0.25))
    model.add(TimeDistributed(Dense(french_vocab_size+1, activation='softmax')))
    
    model.compile(optimizer=Adam(learning_rate), loss = sparse_categorical_crossentropy, metrics = ['accuracy'])
    return model

tmp_x = pad(preproc_english_sentences, max_french_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_french_sentences.shape[-2], 1))

model = bidirectional_model(tmp_x.shape, preproc_french_sentences.shape[1], english_vocab_size, french_vocab_size)
print('model input shape is:-- ', model.input_shape)
print('model output shape is:-- ', model.output_shape)
model.fit(tmp_x, preproc_french_sentences, batch_size=1024, epochs = 10, validation_split=0.30)

model.save('saved_model.yaml')






























    