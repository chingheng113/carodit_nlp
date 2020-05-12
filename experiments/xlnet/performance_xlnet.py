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
#         read_path = os.path.join('results', 'internal', 'round_'+str(n), 'predict_result.pickle')
#         # read_path = os.path.join('results', 'external', 'predict_result_'+str(n)+'.pickle')
#         with open(read_path, 'rb') as file:
#             test_data =pickle.load(file)
#             predict_prob = test_data[in_col + '_pred']
#             true_label = test_data[in_col]
#             fpr, tpr, _ = roc_curve(true_label, predict_prob)
#             roc_auc = auc(fpr, tpr)
#             aucs.append(roc_auc)
#     print(in_col, np.mean(aucs))


# ROC curves should be used when there are roughly equal numbers of observations for each class.
# Precision-Recall curves should be used when there is a moderate to large class imbalance.


# read_path = os.path.join('results', 'internal', 'round_0', 'elapse_time.pickle')
# with open(read_path, 'rb') as file:
#     elapse_time =pickle.load(file)
#     print(elapse_time)



# alone
# read_path = os.path.join('results', 'internal', 'round_1', 'predict_result.pickle')
# # read_path = os.path.join('results_10', 'external', 'predict_result_1.pickle')
# for in_col in in_cols:
#     with open(read_path, 'rb') as file:
#         test_data =pickle.load(file)
#         predict_prob = test_data[in_col + '_pred']
#         true_label = test_data[in_col]
#         fpr, tpr, _ = roc_curve(true_label, predict_prob)
#         roc_auc = auc(fpr, tpr)
#         print(in_col, roc_auc)



# binary
ps = []
for n in range(10):
    # read_path = os.path.join('results_binary', 'internal', 'round_' + str(n), 'predict_result.pickle')
    # read_path = os.path.join('results_binary', 'internal', 'round_0', 'predict_result.pickle')
    read_path = os.path.join('results_binary_2e-5', 'external', 'predict_result_'+str(n)+'.pickle')
    # read_path = os.path.join('results_binary', 'external', 'predict_result_0.pickle')
    with open(read_path, 'rb') as file:
        test_data = pickle.load(file)
        a = test_data[['stenosis', 'stenosis_0_pred', 'stenosis_1_pred']]
        predict_prob = test_data[['stenosis_0_pred', 'stenosis_1_pred']]
        true_label = test_data['stenosis']
        fpr, tpr, _ = roc_curve(true_label, predict_prob['stenosis_1_pred'])
        p = auc(fpr, tpr)
        # print(roc_auc)
        predict_label = predict_prob.values.argmax(axis=1)
        # p = f1_score(true_label, predict_label)
        # p = average_precision_score(true_label, predict_prob['stenosis_1_pred'], average='micro')
        xlnet_precision, xlnet_recall, xlnet_thresholds = precision_recall_curve(true_label, predict_prob['stenosis_1_pred'])
        # p = auc(xlnet_recall, xlnet_precision)
        print(p)
        ps.append(p)
print(round(np.mean(ps), 2), round(np.std(ps), 2))