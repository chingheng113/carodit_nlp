import numpy as np
from sklearn.metrics import precision_recall_curve
from numpy import interp
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from matplotlib import pyplot
import pickle
import os
import pandas as pd

cols = ['RIICA', 'RACA', 'RMCA', 'RPCA', 'RIVA', 'BA', 'LIICA', 'LACA', 'LMCA', 'LPCA', 'LIVA']

fig, axs = pyplot.subplots(3, 4, figsize=(18,12))
for inx, col in enumerate(cols):
    xlnet_precision_array = []
    xlnet_threshold_array = []
    xlnet_mean_recall = np.linspace(0, 1, 100)
    xlnet_auc_array = []

    lstm_precision_array = []
    lstm_threshold_array = []
    lstm_mean_recall = np.linspace(0, 1, 100)
    lstm_auc_array = []

    # rb_precision_array = []
    # rb_threshold_array = []
    # rb_mean_recall = np.linspace(0, 1, 3)
    # rb_auc_array = []

    for n in range(10):
        # xlnet
        xlnet_read_path = os.path.join('xlnet', 'results', 'internal', 'round_' + str(n), 'predict_result.pickle')
        with open(xlnet_read_path, 'rb') as file:
            xlnet_test_data =pickle.load(file)
            xlnet_predict_prob = xlnet_test_data[col + '_pred']
            xlnet_predict_label = np.where(xlnet_predict_prob > 0.5, 1, 0)
            xlnet_true_label = xlnet_test_data[col]
            xlnet_precision, xlnet_recall, xlnet_thresholds = precision_recall_curve(xlnet_true_label, xlnet_predict_prob)
            xlnet_precision_array.append(interp(xlnet_mean_recall, xlnet_precision, xlnet_recall))
            xlnet_pr_auc = auc(xlnet_recall, xlnet_precision)
            xlnet_auc_array.append(xlnet_pr_auc)

        # lstm
        lstm_read_path = os.path.join('lstm', 'results_accuracy', 'internal', 'round_' + str(n), 'predict_result_' + str(n) + '.csv')
        lstm_test_data = pd.read_csv(lstm_read_path)
        lstm_predict_prob = lstm_test_data[col + '_pred']
        lstm_predict_label = np.where(lstm_predict_prob > 0.5, 1, 0)
        lstm_true_label = lstm_test_data[col]
        lstm_precision, lstm_recall, lstm_thresholds = precision_recall_curve(lstm_true_label, lstm_predict_prob)
        lstm_precision_array.append(interp(lstm_mean_recall, lstm_precision, lstm_recall))
        lstm_pr_auc = auc(lstm_recall, lstm_precision)
        lstm_auc_array.append(lstm_pr_auc)

        #RB model
        # result = pd.read_csv(os.path.join('rb', 'internal', 'data_compare_' + str(n) + '.csv'))
        # true_label = result[col]
        # predict_label = result[col + '1']
        # rb_precision, rb_recall, rb_thresholds = precision_recall_curve(true_label, predict_label)
        # rb_precision_array.append(interp(rb_mean_recall, rb_precision, rb_recall))
        # rb_pr_auc = auc(rb_recall, rb_precision)
        # rb_auc_array.append(rb_pr_auc)

    # plot the precision-recall curves
    ax = axs[int(np.floor(inx / 4)), inx % 4]

    xlnet_mean_precision = np.mean(xlnet_precision_array, axis=0)
    xlnet_std_precision = np.std(xlnet_precision_array, axis=0)
    ax.plot(xlnet_mean_recall, xlnet_mean_precision,  label=('XLNet (area = {0:0.2f})'''.format(np.mean(xlnet_auc_array))))
    xlnet_tprs_upper = np.minimum(xlnet_mean_precision + xlnet_std_precision, 1)
    xlnet_tprs_lower = np.maximum(xlnet_mean_precision - xlnet_std_precision, 0)
    ax.fill_between(xlnet_mean_recall, xlnet_tprs_lower, xlnet_tprs_upper, alpha=.2)

    lstm_mean_precision = np.mean(lstm_precision_array, axis=0)
    lstm_std_precision = np.std(lstm_precision_array, axis=0)
    ax.plot(lstm_mean_recall, lstm_mean_precision, label=('LSTM (area = {0:0.2f})'''.format(np.mean(lstm_auc_array))))
    lstm_tprs_upper = np.minimum(lstm_mean_precision + lstm_std_precision, 1)
    lstm_tprs_lower = np.maximum(lstm_mean_precision - lstm_std_precision, 0)
    ax.fill_between(lstm_mean_recall, lstm_tprs_lower, lstm_tprs_upper, alpha=.2)

    # rb_mean_precision = np.mean(rb_precision_array, axis=0)
    # rb_std_precision = np.std(rb_precision_array, axis=0)
    # ax.plot(rb_mean_recall, rb_mean_precision, label=('Rule-based (area = {0:0.2f})'''.format(np.mean(rb_auc_array))))
    # rb_tprs_upper = np.minimum(rb_mean_precision + rb_std_precision, 1)
    # rb_tprs_lower = np.maximum(rb_mean_precision - rb_std_precision, 0)
    # ax.fill_between(rb_mean_recall, rb_tprs_lower, rb_tprs_upper, alpha=.2)


    # axis labels
    ax.set(xlabel='Recall', ylabel='Precision')
    # show the legend
    ax.legend(fontsize='x-small', loc='lower right')
    # show the plot
    ax.set_title(col)

fig.delaxes(axs[2,3])
pyplot.show()

fig.savefig('internalPR.png', dpi=300)