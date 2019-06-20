import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split


data = pd.read_csv('carotid_101518_modified.csv')
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

tokenizer = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;?@[\]^_`{|}~ ')
tokenizer.fit_on_texts(text_arr)
# print(tokenizer.word_counts)
# print( tokenizer.word_index)
# print( tokenizer.word_docs)
t2s = tokenizer.texts_to_sequences(text_arr)
MAX_SENTENCE_LENGTH = len(max(t2s, key=len))
t2s_pad = sequence.pad_sequences(t2s, maxlen=MAX_SENTENCE_LENGTH)

x_train, x_test, Y_train, Y_test = train_test_split(t2s_pad, label_arr, test_size=0.2, random_state=42)



print('done')