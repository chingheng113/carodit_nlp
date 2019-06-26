import pandas as pd
import numpy as np
import os
import sys
import re
import data_util
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
current_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join('/data/linc9/carodit_nlp/')))

# read data
data = pd.read_csv('carotid_101518_modified.csv')
# data = data.loc[0:20, :]
data.dropna(subset=['CONTENT'], axis=0, inplace=True)
# Preprocessing
text_arr = []
label_arr = []
for index, row in data.iterrows():
    label = row[['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA',
                 'LCCA', 'LEICA', 'LIICA', 'LACA', 'LMCA', 'LPCA', 'LEVA', 'LIVA']].values
    corpus = row['CONTENT']
    if '<BASE64>' not in corpus:
        sentences = corpus.split('\n')
        processed_sentence = ''
        for sentence in sentences:
            if len(re.findall(r'[\u4e00-\u9fff]+', sentence)) == 0:
                # no chinese sentence
                if re.search('(>\s*\d+|<\s*\d+)', sentence):
                    sentence = sentence.replace('>', ' greater ')
                    sentence = sentence.replace('<', ' less ')
                sentence = sentence.replace('%', ' percent')
                processed_sentence += sentence+' '
        processed_sentence = re.sub(' +', ' ', processed_sentence)
        text_arr.append(processed_sentence)
        label_arr.append(label)
text_arr = np.array(text_arr)
label_arr = np.array(label_arr)
# Train, Test split
x_train, x_test, Y_train, Y_test = train_test_split(text_arr, label_arr, test_size=0.2, random_state=42)
# tokenize
tokenizer = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(x_train)
t2s_train = tokenizer.texts_to_sequences(x_train)
t2s_test = tokenizer.texts_to_sequences(x_test)
# padding
MAX_SENTENCE_LENGTH = max(len(max(t2s_train, key=len)), len(max(t2s_test, key=len)))
t2s_train_pad = sequence.pad_sequences(t2s_train, maxlen=MAX_SENTENCE_LENGTH)
data_util.save_variable([t2s_train_pad, Y_train], 'training_data.pickle')
t2s_test_pad = sequence.pad_sequences(t2s_test, maxlen=MAX_SENTENCE_LENGTH)
data_util.save_variable([t2s_test_pad, Y_test], 'testing_data.pickle')

# config
config = dict()
config['batch_size'] = 32
config['epochs'] = 50
config['n_hidden'] = 64
config['n_class'] = Y_train.shape[1]
config['input_dim'] = min(2000, len(tokenizer.word_counts))+2
config['output_dim'] = 128
# model
model = Sequential()
model.add(Embedding(input_dim=config['input_dim'], output_dim=config['output_dim'], input_length=MAX_SENTENCE_LENGTH))
model.add(LSTM(config['n_hidden'], dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(config['n_class']))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(t2s_train_pad, Y_train,
                    batch_size=config['batch_size'],
                    epochs=config['epochs'],
                    validation_split=0.2,
                    callbacks=[
                        ReduceLROnPlateau(factor=0.5, patience=int(config['epochs']/10), verbose=1),
                        EarlyStopping(verbose=1, patience=int(config['epochs']/2)),
                        ModelCheckpoint(os.path.join(current_path, model.name + '.h5'), save_best_only=True, verbose=1)
                    ])
# History
data_util.save_variable(history.history, 'history.pickle')
# result
y_pred_p = model.predict(t2s_test_pad)
result = np.concatenate((y_pred_p, Y_test), axis=1)
data_util.save_variable(result, 'predict_y.pickle')
print('done')