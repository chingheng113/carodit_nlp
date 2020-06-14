import numpy as np
from sklearn.metrics import precision_recall_curve
from scipy import interp
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

    clinical_xlnet_precision_array = []
    clinical_xlnet_threshold_array = []
    clinical_xlnet_mean_recall = np.linspace(0, 1, 100)
    clinical_xlnet_auc_array = []

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

            # clinical_xlnet
            clinical_xlnet_read_path = os.path.join('clinical_xlnet', 'results', 'internal', 'round_' + str(n), 'predict_result.pickle')
            # clinical_xlnet_read_path = os.path.join('clinical_xlnet', 'results_1e-5', 'internal', 'round_' + str(n), 'predict_result.pickle')

            with open(clinical_xlnet_read_path, 'rb') as file:
                clinical_xlnet_test_data = pickle.load(file)
                clinical_xlnet_predict_prob = clinical_xlnet_test_data[col + '_pred']
                clinical_xlnet_predict_label = np.where(clinical_xlnet_predict_prob > 0.5, 1, 0)
                clinical_xlnet_true_label = clinical_xlnet_test_data[col]
                clinical_xlnet_precision, clinical_xlnet_recall, clinical_xlnet_thresholds = precision_recall_curve(clinical_xlnet_true_label,
                                                                                         clinical_xlnet_predict_prob)
                clinical_xlnet_precision_array.append(interp(clinical_xlnet_mean_recall, clinical_xlnet_precision, clinical_xlnet_recall))
                clinical_xlnet_pr_auc = auc(clinical_xlnet_recall, clinical_xlnet_precision)
                clinical_xlnet_auc_array.append(clinical_xlnet_pr_auc)

    # plot the precision-recall curves
    xlnet_no_skill = len(xlnet_true_label[xlnet_true_label==1]) / len(xlnet_true_label)
    ax = axs[int(np.floor(inx/4)), inx%4]
    ax.plot([0, 1], [xlnet_no_skill, xlnet_no_skill], linestyle='--', label='No Skill')

    xlnet_mean_precision = np.mean(xlnet_precision_array, axis=0)
    xlnet_std_precision = np.std(xlnet_precision_array, axis=0)
    ax.plot(xlnet_mean_recall, xlnet_mean_precision,  label=('XLNet (area = {0:0.2f})'''.format(np.mean(xlnet_auc_array))))

    clinical_xlnet_mean_precision = np.mean(clinical_xlnet_precision_array, axis=0)
    clinical_xlnet_std_precision = np.std(clinical_xlnet_precision_array, axis=0)
    ax.plot(clinical_xlnet_mean_recall, clinical_xlnet_mean_precision, label=('clinicalXLNet (area = {0:0.2f})'''.format(np.mean(clinical_xlnet_auc_array))))

    xlnet_tprs_upper = np.minimum(xlnet_mean_precision + xlnet_std_precision, 1)
    xlnet_tprs_lower = np.maximum(xlnet_mean_precision - xlnet_std_precision, 0)
    ax.fill_between(xlnet_mean_recall, xlnet_tprs_lower, xlnet_tprs_upper, alpha=.2)

    clinical_xlnet_tprs_upper = np.minimum(clinical_xlnet_mean_precision + clinical_xlnet_std_precision, 1)
    clinical_xlnet_tprs_lower = np.maximum(clinical_xlnet_mean_precision - clinical_xlnet_std_precision, 0)
    ax.fill_between(clinical_xlnet_mean_recall, clinical_xlnet_tprs_lower, clinical_xlnet_tprs_upper, alpha=.2)

    # axis labels
    ax.set(xlabel='Recall', ylabel='Precision')
    # show the legend
    ax.legend(fontsize='x-small', loc='lower right')
    # show the plot
    ax.set_title(col)

fig.delaxes(axs[2,3])
pyplot.show()

fig.savefig('inPR.png', dpi=300)