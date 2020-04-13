import pandas as pd
import os
import argparse
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import pickle
import time

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                        default=None,
                        type=int,
                        required=True,
                        help="batch size")
    parser.add_argument("--epochs",
                        default=None,
                        type=int,
                        required=True,
                        help="epochs")
    parser.add_argument("--n_hidden",
                        default=None,
                        type=int,
                        required=True,
                        help="n_hidden")
    parser.add_argument("--round",
                        default=None,
                        type=str,
                        required=True,
                        help="Which data round will be using")
    return parser


# def text_preprocessing():
#     text_arr = []
#     label_arr = []
#     for index, row in data.iterrows():
#         label = row[['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA',
#                      'LCCA', 'LEICA', 'LIICA', 'LACA', 'LMCA', 'LPCA', 'LEVA', 'LIVA']].values
#         corpus = row['CONTENT']
#         if '<BASE64>' not in corpus:
#             sentences = corpus.split('\n')
#             processed_sentence = ''
#             for sentence in sentences:
#                 if len(re.findall(r'[\u4e00-\u9fff]+', sentence)) == 0:
#                     # no chinese sentence
#                     if re.search('(>\s*\d+|<\s*\d+)', sentence):
#                         sentence = sentence.replace('>', ' greater ')
#                         sentence = sentence.replace('<', ' less ')
#                     sentence = sentence.replace('%', ' percent')
#                     processed_sentence += sentence + ' '
#             processed_sentence = re.sub(' +', ' ', processed_sentence)
#             text_arr.append(processed_sentence)
#             label_arr.append(label)
#     text_arr = np.array(text_arr)

def read_internal_data(n):
    train_data = pd.read_csv(os.path.join('..', 'data', 'internal', 'round_'+n, 'training_'+n+'.csv'))
    label_cols = list(train_data.columns)[2:-1]
    test_data = pd.read_csv(os.path.join('..', 'data', 'internal', 'round_'+n, 'testing_'+n+'.csv'))
    return train_data, test_data, label_cols


def read_external_data(n):
    test_data = pd.read_csv(os.path.join('..', 'data', 'external', 'testing.csv'))
    label_cols = list(test_data.columns)[2:-1]
    return test_data, label_cols


def model_training(config, train, label_cols, round_n):
    x_train = train['processed_content']
    y_train = train[label_cols]
    # tokenize
    tokenizer = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(x_train)
    t2s_train = tokenizer.texts_to_sequences(x_train)
    # padding
    # max(len(max(t2s_train, key=len)), len(max(t2s_test, key=len)))
    MAX_SENTENCE_LENGTH = len(max(t2s_train, key=len))
    t2s_train_pad = sequence.pad_sequences(t2s_train, maxlen=MAX_SENTENCE_LENGTH)
    # data_util.save_variable([t2s_train_pad, Y_train], 'training_data.pickle')
    config['n_class'] = len(label_cols)
    config['input_dim'] = len(tokenizer.word_counts)+2
    config['output_dim'] = 128
    start = time.time()
    # model
    model = Sequential(name='lstm_'+round_n)
    model.add(Embedding(input_dim=config['input_dim'], output_dim=config['output_dim'], input_length=MAX_SENTENCE_LENGTH))
    model.add(LSTM(config['n_hidden'], dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(int(config['n_hidden']/2), dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(config['n_class']))
    model.add(Activation("sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(t2s_train_pad, y_train,
                        batch_size=config['batch_size'],
                        epochs=config['epochs'],
                        validation_split=0.2,
                        callbacks=[
                            ReduceLROnPlateau(factor=0.5, patience=int(config['epochs']/10), verbose=1),
                            EarlyStopping(verbose=1, patience=int(config['epochs']/10)),
                            ModelCheckpoint(os.path.join('models', model.name+'.h5'),
                                            save_best_only=True, verbose=1)
                        ])
    end = time.time()
    elapse = end - start
    # History
    with open(os.path.join('results', 'internal', 'round_'+round_n, 'history_'+round_n+'.pickle'), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    with open(os.path.join('results', 'internal', 'round_'+round_n, 'elapse_time.pickle'), 'wb') as file_pi:
        pickle.dump(elapse, file_pi)

    return tokenizer, MAX_SENTENCE_LENGTH, model


def model_testing(testing, label_cols, model, tokenizer, max_len, round_n, ex_in):
    x_test = testing['processed_content']
    y_test = testing[label_cols]
    # tokenize
    t2s_test = tokenizer.texts_to_sequences(x_test)
    # padding
    t2s_test_pad = sequence.pad_sequences(t2s_test, maxlen=max_len)
    # data_util.save_variable([t2s_test_pad, Y_test], 'testing_data.pickle')
    y_pred_p = model.predict(t2s_test_pad)
    for index, elem in enumerate(label_cols):
        testing[elem + '_pred'] = y_pred_p[:, index]
    if ex_in == 'internal':
        testing.to_csv(os.path.join('results', 'internal', 'round_'+round_n, 'predict_y_'+round_n+'.csv'), index=False)
    else:
        testing.to_csv(os.path.join('results', 'external', 'predict_y_'+round_n+'.csv'), index=False)
    print('testing done')


if __name__ == '__main__':
    parser = setup_parser()
    args = parser.parse_args()
    round_nm = args.round
    # config
    configuration = dict()
    configuration['batch_size'] = args.batch_size
    configuration['epochs'] = args.epochs
    configuration['n_hidden'] = args.n_hidden

    in_train_data, in_test_data, in_label_cols = read_internal_data(round_nm)
    # in_train_data = in_train_data.head(10)
    trained_tokenizer, max_len, trained_model = model_training(configuration, in_train_data, in_label_cols, round_nm)
    model_testing(in_test_data, in_label_cols, trained_model, trained_tokenizer, max_len, round_nm, 'internal')

    ex_test_data, ex_label_cols = read_external_data(round_nm)
    model_testing(ex_test_data, ex_label_cols, trained_model, trained_tokenizer, max_len, round_nm, 'external')

