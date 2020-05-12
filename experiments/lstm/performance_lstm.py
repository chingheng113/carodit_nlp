from sklearn.metrics import roc_curve, auc
from sklearn.metrics import f1_score, matthews_corrcoef, average_precision_score, precision_recall_curve
import pandas as pd
import numpy as np
import pickle
import os


in_cols = ['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA', 'LCCA', 'LEICA', 'LIICA',
           'LACA', 'LMCA', 'LPCA', 'LEVA', 'LIVA']


# for in_col in in_cols:
#     aucs = []
#     for n in range(10):
#         # read_path = os.path.join('results', 'internal', 'round_' + str(n), 'predict_result_' + str(n) + '.csv')
#         read_path = os.path.join('results', 'external', 'predict_result_' + str(n) + '.csv')
#         # read_path = os.path.join('results_old', 'internal', 'round_' + str(n), 'predict_result_' + str(n) + '.csv')
#         test_data = pd.read_csv(read_path)
#         predict_prob = test_data[in_col+'_pred']
#         true_label = test_data[in_col]
#         fpr, tpr, _ = roc_curve(true_label, predict_prob)
#         roc_auc = auc(fpr, tpr)
#         aucs.append(roc_auc)
#     print(in_col, np.mean(aucs))
#
#
#
#
#
# read_path = os.path.join('results', 'internal', 'round_0', 'elapse_time.pickle')
# with open(read_path, 'rb') as file:
#     elapse_time =pickle.load(file)
#     print(elapse_time)


# binary
ps = []
for n in range(10):
    read_path = os.path.join('results_binary', 'internal', 'round_' + str(n), 'predict_result_' + str(n) + '.csv')
    #         read_path = os.path.join('results_binary', 'external', 'predict_result_' + str(n) + '.csv')
    test_data = pd.read_csv(read_path)
    predict_prob = test_data['stenosis_pred']
    true_label = test_data['stenosis']
    fpr, tpr, _ = roc_curve(true_label, predict_prob)
    roc_auc = auc(fpr, tpr)
    # print(roc_auc)
    predict_label = np.where(predict_prob > 0.5, 1, 0)
    # p = f1_score(true_label, predict_label)
    # p = average_precision_score(true_label, predict_prob, average='micro')
    xlnet_precision, xlnet_recall, xlnet_thresholds = precision_recall_curve(true_label, predict_prob)
    p = auc(xlnet_recall, xlnet_precision)
    print(p)
    ps.append(p)
print(round(np.mean(ps), 2), round(np.std(ps), 2))