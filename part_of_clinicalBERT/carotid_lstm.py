import numpy as np
import os
import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.layers.core import Activation, Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
import argparse
current_path = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join('/data/linc9/carodit_nlp/')))
sys.path.append('../')
from carotid_data import data_util

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--round",
                        default=None,
                        type=str,
                        required=True,
                        help="Which data round will be using")
    return parser


def make_one_line(data):
    text_arr = []
    for row in data:
        sentences = row.split('\n')
        processed_sentence = ''
        for sentence in sentences:
            processed_sentence += sentence + ' '
        text_arr.append(processed_sentence)
    return np.array(text_arr)


def main():
    parser = setup_parser()
    args = parser.parse_args()
    round_nm = args.round
    # read data
    training_data = data_util.read_variable(os.path.join('round_'+round_nm, 'training_bert.pickle'))
    x_train = training_data['processed_content']
    x_train = make_one_line(x_train)
    Y_train = training_data[['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA', 'LCCA', 'LEICA', 'LIICA',
                            'LACA', 'LMCA', 'LPCA', 'LEVA', 'LIVA']].values

    test_data = data_util.read_variable(os.path.join('round_'+round_nm, 'test_bert.pickle'))
    x_test = test_data['processed_content']
    x_test = make_one_line(x_test)
    Y_test = test_data[['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA', 'LCCA', 'LEICA', 'LIICA',
                        'LACA', 'LMCA', 'LPCA', 'LEVA', 'LIVA']].values

    # tokenize
    tokenizer = Tokenizer(num_words=None)
    tokenizer.fit_on_texts(x_train)
    t2s_train = tokenizer.texts_to_sequences(x_train)
    t2s_test = tokenizer.texts_to_sequences(x_test)
    # padding
    MAX_SENTENCE_LENGTH = max(len(max(t2s_train, key=len)), len(max(t2s_test, key=len)))
    t2s_train_pad = sequence.pad_sequences(t2s_train, maxlen=MAX_SENTENCE_LENGTH)
    # data_util.save_variable([t2s_train_pad, Y_train], 'training_data.pickle')
    t2s_test_pad = sequence.pad_sequences(t2s_test, maxlen=MAX_SENTENCE_LENGTH)
    # data_util.save_variable([t2s_test_pad, Y_test], 'testing_data.pickle')

    # config
    config = dict()
    config['batch_size'] = 32
    config['epochs'] = 50
    config['n_hidden'] = 64
    config['n_class'] = Y_train.shape[1]
    config['input_dim'] = max(2000, len(tokenizer.word_counts))+2
    config['output_dim'] = 128
    # model
    model = Sequential(name='lstm_'+round_nm)
    model.add(Embedding(input_dim=config['input_dim'], output_dim=config['output_dim'], input_length=MAX_SENTENCE_LENGTH))
    model.add(LSTM(config['n_hidden'], dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(int(config['n_hidden']/2), dropout=0.2, recurrent_dropout=0.2))
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
    data_util.save_variable(history.history, 'history_'+round_nm+'.pickle')
    # result
    y_pred_p = model.predict(t2s_test_pad)
    result = np.concatenate((y_pred_p, Y_test), axis=1)
    data_util.save_variable(result, 'predict_y_'+round_nm+'.pickle')
    print('done')


if __name__ == '__main__':
    main()