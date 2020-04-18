from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
import pickle
import os


in_cols = ['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA', 'LCCA', 'LEICA', 'LIICA',
           'LACA', 'LMCA', 'LPCA', 'LEVA', 'LIVA']


for in_col in in_cols:
    aucs = []
    for n in range(10):
        # read_path = os.path.join('results', 'internal', 'round_' + str(n), 'predict_result_' + str(n) + '.csv')
        read_path = os.path.join('results', 'external', 'predict_result_' + str(n) + '.csv')
        # read_path = os.path.join('results_old', 'internal', 'round_' + str(n), 'predict_result_' + str(n) + '.csv')
        test_data = pd.read_csv(read_path)
        predict_prob = test_data[in_col+'_pred']
        true_label = test_data[in_col]
        fpr, tpr, _ = roc_curve(true_label, predict_prob)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    print(in_col, np.mean(aucs))





read_path = os.path.join('results', 'internal', 'round_0', 'elapse_time.pickle')
with open(read_path, 'rb') as file:
    elapse_time =pickle.load(file)
    print(elapse_time)