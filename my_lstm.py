import pandas as pd
import numpy as np
import pickle
import os
current_path = os.path.dirname(__file__)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential

data = pd.read_csv('carotid_101518_modified.csv')
# data = data.loc[0:20, :]
data.dropna(subset=['CONTENT'], axis=0, inplace=True)
text_arr = []
label_arr = []
for index, row in data.iterrows():
    label = row[['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA',
                 'LCCA', 'LEICA', 'LIICA', 'LACA', 'LMCA', 'LPCA', 'LEVA', 'LIVA']].values
    doc = row['CONTENT']
    doc = doc.replace("\n", " ")
    if '<BASE64>' not in doc:
        text_arr.append(doc)
        label_arr.append(label)
text_arr = np.array(text_arr)
label_arr = np.array(label_arr)
n_class = label_arr[0].shape[0]

tokenizer = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(text_arr)
vocab_size = len(tokenizer.word_counts)+2
# print( tokenizer.word_index)
# print( tokenizer.word_docs)
t2s = tokenizer.texts_to_sequences(text_arr)
MAX_SENTENCE_LENGTH = len(max(t2s, key=len))
t2s_pad = sequence.pad_sequences(t2s, maxlen=MAX_SENTENCE_LENGTH)

x_train, x_test, Y_train, Y_test = train_test_split(t2s_pad, label_arr, test_size=0.2, random_state=42)

model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=MAX_SENTENCE_LENGTH))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(n_class))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, Y_train, batch_size=32, epochs=50, validation_split=0.2)
y_pred_p = model.predict(x_test)

with open(os.path.join(current_path, 'predict_y.pickle'), 'wb') as file_pi:
    result = np.concatenate((y_pred_p, Y_test), axis=1)
    pickle.dump(result, file_pi)
print('done')