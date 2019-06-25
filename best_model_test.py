import pandas as pd
import numpy as np
import pickle
import os
current_path = os.path.dirname(__file__)
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.models import load_model


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

# tokenize
tokenizer = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(text_arr)
n_words = len(tokenizer.word_counts)
# print( tokenizer.word_index)
# print( tokenizer.word_docs)
t2s = tokenizer.texts_to_sequences(text_arr)
MAX_SENTENCE_LENGTH = len(max(t2s, key=len))
t2s_pad = sequence.pad_sequences(t2s, maxlen=MAX_SENTENCE_LENGTH)

# Train, Test split
x_train, x_test, Y_train, Y_test = train_test_split(t2s_pad, label_arr, test_size=0.2, random_state=42)

model = load_model('sequential_1.h5')
y_pred_p = model.predict(x_test)

print('done')