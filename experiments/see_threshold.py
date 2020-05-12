import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import auc
from matplotlib import pyplot
import pickle
import os

# cols = ['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA', 'LCCA', 'LEICA', 'LIICA',
#            'LACA', 'LMCA', 'LPCA', 'LEVA', 'LIVA']

cols = ['BA']

print('xlnet')
for col in cols:
    for t in np.arange(0.0, 1.0, 0.1):
        lr_f1s, lr_ms, lr_prs, lr_res = [], [], [], []
        for n in range(10):
            # read_path = os.path.join('xlnet', 'results', 'internal', 'round_' + str(n), 'predict_result.pickle')
            read_path = os.path.join('xlnet', 'results', 'external', 'predict_result_' + str(n) + '.pickle')
            with open(read_path, 'rb') as file:
                test_data = pickle.load(file)
                predict_prob = test_data[col + '_pred']
                predict_label = np.where(predict_prob > t, 1, 0)
                true_label = test_data[col]

                precision, recall, fscore, support = precision_recall_fscore_support(true_label, predict_label)
                lr_prs.append(precision)
                lr_res.append(recall)

                lr_precision, lr_recall, _ = precision_recall_curve(true_label, predict_prob)
                f1 = f1_score(true_label, predict_label, average='macro')
                lr_f1s.append(f1)

                ms = matthews_corrcoef(true_label, predict_label)
                lr_ms.append(ms)

        print(t, np.mean(lr_prs), np.mean(lr_res))

