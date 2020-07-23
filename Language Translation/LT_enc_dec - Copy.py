#import all the required libraries
import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, LSTM, Input, Activation, concatenate, dot, TimeDistributed, Lambda
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
from keras.losses import categorical_crossentropy
from string import punctuation
import re
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#reading the files
def read_data():
    with open('data/small_vocab_en', 'r', encoding = 'utf-8') as f:
        file_english = f.readlines()
    with open('data/small_vocab_fr', 'r', encoding = 'utf-8') as g:
        file_french = g.readlines()
    
    english_sentences = [sentences[:-2].strip() for sentences in file_english]
    french_sentences = [sentences[:-2].strip() for sentences in file_french]

    return english_sentences, french_sentences


english_sentences, french_sentences = read_data()

#creating a dataframe, we do this for easy and fast processing of data
df = pd.DataFrame()
df['English'] = english_sentences
df['French'] = french_sentences
df = shuffle(df)

def data_preprocess(df):
    df['English']=df['English'].apply(lambda x: x.lower())
    df['French'] = df['French'].apply(lambda x: x.lower())
    
    # Remove quotes
    df['English']=df['English'].apply(lambda x: re.sub("'", '', x))
    df['French']=df['French'].apply(lambda x: re.sub("'", '', x))
    
    # Remove all the special characters
    exclude = set(punctuation) # Set of all special characters
    df['English']=df['English'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    df['French']=df['French'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
    
    #removing extra white spaces
    df['English']=df['English'].apply(lambda x: x.strip())
    df['French']=df['French'].apply(lambda x: x.strip())
    
    #using start and end tokens to ensure the start and end of sentence
    df['French'] = df['French'].apply(lambda x : '<START> '+ x + ' <END>')
    
    return df

df = data_preprocess(df)

#creating the vocabulary for english and french sentences
def create_vocab(df):
    #taking empty sets for english and french words
    eng_words = set()
    fr_words = set()
    
    for eng in df['English']:
        for word in eng.split():
            if word not in eng_words:
                eng_words.add(word)
    for fr in df['French']:
        for word in fr.split():
            if word not in fr_words:
                fr_words.add(word)
                
    return sorted(list(eng_words)), sorted(list(fr_words))

eng_vocab, fr_vocab = create_vocab(df)

max_eng_sent_length = np.max([len(sent.split()) for sent in df['English']])
max_fr_sent_length =  np.max([len(sent.split()) for sent in df['French']])

num_encoder_tokens = len(eng_vocab)
num_decoder_tokens = len(fr_vocab)
#Add +1 for 0 padding(padding used to make all sentences equal in length)
num_decoder_tokens += 1

#create word -> token and token -> word dictionary
eng_word_to_token = {word:i+1 for i, word in enumerate(eng_vocab)} 
eng_token_to_word = {i:word for word, i in eng_word_to_token.items()}
fr_word_to_token  = {word:i+1 for i, word in enumerate(fr_vocab)}
fr_token_to_word  = {i:word for word, i in fr_word_to_token.items()}



#creating a generator
def generate_batch(x, y, batch_size=128):
    
    while True:
        for j in range(0, len(x), batch_size):
            #this is a bit explicit code, we can also use keras tokenizer and padding
            encoder_input_data = np.zeros((batch_size, max_eng_sent_length))
            decoder_input_data = np.zeros((batch_size, max_fr_sent_length))
            
            decoder_output_data = np.zeros((batch_size, max_fr_sent_length, num_decoder_tokens))
            
            for i, (input_text, output_text) in enumerate(zip(x[j:j+batch_size], y[j:j+batch_size])):
                for t, word in enumerate(input_text.split()):
                    encoder_input_data[i, t] = eng_word_to_token[word]
                for t, word in enumerate(output_text.split()):
                    if t<len(output_text.split())-1:
                        decoder_input_data[i, t] = fr_word_to_token[word]
                    if t>0:
                        decoder_output_data[i, t-1, fr_word_to_token[word]] = 1
            yield ([encoder_input_data, decoder_input_data], decoder_output_data)


X = df['English']
y = df['French']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)




#defining the encoder training model
hidden_dims = 200
encoder_input = Input(shape = (None,))
enc_embed_layer = Embedding(num_encoder_tokens, hidden_dims, mask_zero = True)
enc_embeds = enc_embed_layer(encoder_input)
enc_lstm = LSTM(hidden_dims, return_state = True)
encoder_outputs, state_h, state_c = enc_lstm(enc_embeds)
encoder_states = [state_h, state_c]

#defining decoder training model
hidden_dims = 200
decoder_input = Input(shape = (None,))
dec_embed_layer = Embedding(num_decoder_tokens, hidden_dims, mask_zero = True)
dec_embeds = dec_embed_layer(decoder_input)
dec_lstm = LSTM(hidden_dims, return_sequences = True, return_state = True)
decoder_outputs, h, c = dec_lstm(dec_embeds, initial_state = encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation = 'softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_input, decoder_input], decoder_outputs)

model.compile(optimizer = Adam(learning_rate = 0.01), loss = 'categorical_crossentropy', metrics = ['acc'])

batch_size = 128
model.fit_generator(generator = generate_batch(X_train, y_train, batch_size = batch_size),
                                               steps_per_epoch = len(X_train)//batch_size,
                                               epochs = 50,
                                               validation_data = generate_batch(X_test, y_test,
                                                                                batch_size = batch_size),
                                               validation_steps = len(X_test)//batch_size)

#defining decoder training model
# hidden_dims = 200
# decoder_input = Input(shape = (max_fr_sent_length,))
# dec_embed_layer = Embedding(num_decoder_tokens, hidden_dims, input_length = max_fr_sent_length, mask_zero = True)
# dec_embeds = dec_embed_layer(decoder_input)
# dec_lstm = LSTM(hidden_dims, return_sequences = True, return_state = True)
# decoder_outputs, h, c = dec_lstm(dec_embeds)

########TRYIIIIIINNNNGGGGGGG..........................................................
# hidden_dims = 200
# decoder_input = Input(shape = (max_fr_sent_length,))
# dec_embed_layer = Embedding(num_decoder_tokens, hidden_dims, input_length = max_fr_sent_length, mask_zero = True)
# dec_embeds = dec_embed_layer(decoder_input)
# dec_lstm = LSTM(hidden_dims, return_sequences = True, return_state = True)
# decoder_outputs, h, c = dec_lstm(dec_embeds)


# wa_dot_hs = Dense(hidden_dims, use_bias = False)(encoder_outputs)
# h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_dims,))(decoder_outputs)

# attention = dot([wa_dot_hs, h_t], axes = (2,1))
# attention = Activation('softmax')(attention)

# context = dot([attention, encoder_outputs], axes = (1,1))
# pre_activation = concatenate([context, h_t])
# outputs = Dense(len(fr_vocab), use_bias = False, activation = 'tanh')(pre_activation)
# outputs = Dense(len(fr_vocab), use_bias = False, activation = 'softmax')(outputs)



model.save_weights('nmt_weights.h5')

#load the weights in case you have not trained the model
model.load_weights('nmt_weights.h5')


#defining decoder inference model
# encoder_model = Model(encoder_input, encoder_states)

# decoder_state_h = Input(shape = (hidden_dims,))
# decoder_state_c = Input(shape = (hidden_dims,))
# decoder_state_inputs = [decoder_state_h, decoder_state_c]
# decoder_embed = dec_embed_layer(decoder_input)
# decoder_outputs2, state_h2, state_c2 = dec_lstm(decoder_embed, initial_state = decoder_state_inputs)
# decoder_states2 = [state_h2, state_c2]
# decoder_outputs2 = decoder_dense(decoder_outputs2)

# decoder_model = Model([decoder_input] + decoder_state_inputs, [decoder_outputs2] + decoder_states2)


# def decode_seq(input_seq):
#     state_vector = encoder_model.predict(input_seq)
#     target_seq = np.zeros((1,1))
#     target_seq[0, 0] = fr_word_to_token['<START>']
#     print(type(target_seq))
#     print(type(state_vector))
#     stop_condition = False
#     decoded_sentence = ''
#     while not stop_condition:
#         output_tokens, h, c = decoder_model.predict([target_seq] + state_vector)
#         #It's for excluding the END clause
#         decoded_output = np.argmax(output_tokens[0, -1, :])
#         decoded_word = fr_token_to_word[decoded_output]
#         if decoded_word != '<END>':
#             decoded_sentence += ' ' + decoded_word
        
#         if decoded_word == '<END>':
#             stop_condition = True
        
#         target_seq = np.zeros((1, 1))
#         target_seq[0, 0] =  decoded_output
        
#         state_vector = [h, c]
#     return decoded_sentence
    
# data_gen = generate_batch(X_train, y_train, batch_size = 1)
# (input_data, actual_output), _ = next(data_gen)

# result = decode_seq(input_data)

# print('Input Sentence: ', X_train[1])
# print('Output Sentence: ', y_train[1])
# print('Predicted Sentence', result)

                     
xx = np.random.randint(1, 10, (2,3,3))
    
    
    
    
    
    