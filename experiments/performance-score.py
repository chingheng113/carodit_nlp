import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import precision_recall_fscore_support, average_precision_score



# Compute average precision (AP) from prediction scores
# AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight:

cols = ['RIICA', 'RACA', 'RMCA', 'RPCA', 'RIVA', 'BA', 'LIICA', 'LACA', 'LMCA', 'LPCA', 'LIVA']

print('xlnet')
for col in cols:
    ps = []
    lr_prs, lr_res = [], []
    for n in range(10):
        # read_path = os.path.join('xlnet', 'results', 'internal', 'round_' + str(n), 'predict_result.pickle')
        read_path = os.path.join('xlnet', 'results', 'external', 'predict_result_'+ str(n) +'.pickle')
        with open(read_path, 'rb') as file:
            test_data =pickle.load(file)
            predict_prob = test_data[col + '_pred']
            predict_label = np.where(predict_prob > 0.5, 1, 0)
            true_label = test_data[col]
            # p = matthews_corrcoef(true_label, predict_label)
            # p = f1_score(true_label, predict_label, average='binary')
            p = average_precision_score(true_label, predict_prob, average='micro')
            ps.append(p)
            precision, recall, fscore, support = precision_recall_fscore_support(true_label, predict_label)
            lr_prs.append(precision)
            lr_res.append(recall)
    print(col, round(np.mean(ps), 2), round(np.std(ps), 2))
    # print(col, round(np.mean(lr_prs), 2), round(np.std(lr_res), 2))


print('lstm')
for col in cols:
    ps = []
    for n in range(10):
        # read_path = os.path.join('lstm', 'results_accuracy', 'internal', 'round_' + str(n), 'predict_result_' + str(n) + '.csv')
        read_path = os.path.join('lstm', 'results_accuracy', 'external', 'predict_result_' + str(n) + '.csv')
        test_data = pd.read_csv(read_path)
        predict_prob = test_data[col + '_pred']
        predict_label = np.where(predict_prob > 0.5, 1, 0)
        true_label = test_data[col]
        # p = matthews_corrcoef(true_label, predict_label)
        # p = f1_score(true_label, predict_label, average='micro')
        p = average_precision_score(true_label, predict_prob, average='micro')
        ps.append(p)
    print(col, round(np.mean(ps), 2), round(np.std(ps), 2))
    # print(col, round(np.mean(lr_prs), 2), round(np.std(lr_res), 2))


print('clinical xlnet')
for col in cols:
    ps = []
    lr_prs, lr_res = [], []
    for n in range(10):
        read_path = os.path.join('clinical_xlnet', 'results', 'internal', 'round_' + str(n), 'predict_result.pickle')
        # read_path = os.path.join('clinical_xlnet', 'results', 'external', 'predict_result_'+ str(n) +'.pickle')
        with open(read_path, 'rb') as file:
            test_data =pickle.load(file)
            predict_prob = test_data[col + '_pred']
            predict_label = np.where(predict_prob > 0.5, 1, 0)
            true_label = test_data[col]
            # p = matthews_corrcoef(true_label, predict_label)
            # p = f1_score(true_label, predict_label, average='binary')
            p = average_precision_score(true_label, predict_prob, average='micro')
            ps.append(p)
            precision, recall, fscore, support = precision_recall_fscore_support(true_label, predict_label)
            lr_prs.append(precision)
            lr_res.append(recall)
    print(col, round(np.mean(ps), 2), round(np.std(ps), 2))
    # print(col, round(np.mean(lr_prs), 2), round(np.std(lr_res), 2))


print('RB model')
for col in cols:
    ps = []
    lr_prs, lr_res = [], []
    for n in range(10):
        # result = pd.read_csv(os.path.join('rb','internal', 'data_compare_' + str(n) + '.csv'))
        result = pd.read_csv(os.path.join('rb','external.csv'))
        true_label = result[col]
        predict_label = result[col + '1']
        # p = matthews_corrcoef(true_label, predict_label)
        # p = f1_score(true_label, predict_label, average='binary')
        p = average_precision_score(true_label, predict_label, average='micro')
        ps.append(p)
        precision, recall, fscore, support = precision_recall_fscore_support(true_label, predict_label)
        lr_prs.append(precision)
        lr_res.append(recall)
    print(col, round(np.mean(ps), 2), round(np.std(ps), 2))
    # print(col, round(np.mean(lr_prs), 2), round(np.std(lr_res), 2))


# Finally, 'macro' calculates the F1 separated by class but not using weights for the aggregation:
# 𝐹1𝑐𝑙𝑎𝑠𝑠1+𝐹1𝑐𝑙𝑎𝑠𝑠2+⋅⋅⋅+𝐹1𝑐𝑙𝑎𝑠𝑠𝑁
# which resuls in a bigger penalisation when your model does not perform well with the minority classes.