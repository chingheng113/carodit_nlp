import pandas as pd
import numpy as np
import pickle
import os
current_path = os.path.dirname(__file__)
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt


def plot_training_acc():
    with open('history_binary.pickle', 'rb') as f:
        history = pickle.load(f)
        plt.plot(history['acc'])
        plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper left')
        plt.show()


def plot_training_loss():
    with open('history_binary.pickle', 'rb') as f:
        history = pickle.load(f)
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['training', 'validation'], loc='upper left')
        plt.show()


with open('predict_y_binary.pickle', 'rb') as file:
    result =pickle.load(file)
    predict_y_p = result[:,0:2]
    true_y = result[:,2:].astype(int)
    # true_y = (true_y == 1)
    plot_training_acc()
    plot_training_loss()
    for i in range(0, 2):
        p_label = predict_y_p[:, i]
        p_label_b = (p_label > 0.5).astype(int)
        t_label = true_y[:, i]
        fpr, tpr, thresholds = roc_curve(t_label, p_label)
        roc_auc = auc(fpr, tpr)
        print('===', i)
        print('AUC = ', round(roc_auc, 4))
        # plt.plot(fpr, tpr, label='(AUC = %0.3f )' % roc_auc, lw=1, alpha=.8)
        print(classification_report(t_label, p_label_b, digits=4))
        print(confusion_matrix(t_label, p_label_b))
    print('done')