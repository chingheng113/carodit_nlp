import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import f1_score, matthews_corrcoef, roc_curve, precision_recall_curve, auc, roc_auc_score
from sklearn.metrics import precision_recall_fscore_support, average_precision_score



# Compute average precision (AP) from prediction scores
# AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight:


print('xlnet')
ps = []
for n in range(10):
    read_path = os.path.join('xlnet', 'results_binary', 'internal', 'round_' + str(n), 'predict_result.pickle')
    # read_path = os.path.join('xlnet', 'results_binary', 'external', 'predict_result_'+str(n)+'.pickle')
    with open(read_path, 'rb') as file:
        test_data = pickle.load(file)
        a = test_data[['stenosis', 'stenosis_0_pred', 'stenosis_1_pred']]
        predict_prob = test_data[['stenosis_0_pred', 'stenosis_1_pred']]
        true_label = test_data['stenosis']
        fpr, tpr, _ = roc_curve(true_label, predict_prob['stenosis_1_pred'])
        # p = auc(fpr, tpr)
        # print(roc_auc)
        predict_label = predict_prob.values.argmax(axis=1)
        # p = f1_score(true_label, predict_label)
        # p = average_precision_score(true_label, predict_prob['stenosis_1_pred'], average='micro')
        # xlnet_precision, xlnet_recall, xlnet_thresholds = precision_recall_curve(true_label, predict_prob['stenosis_1_pred'])
        # p = auc(xlnet_recall, xlnet_precision)
        p = roc_auc_score(true_label, predict_prob['stenosis_1_pred'])
        print(p)
        ps.append(p)
print(round(np.mean(ps), 2), round(np.std(ps), 2))
#

# print('lstm')
# ps = []
# for n in range(10):
#     # read_path = os.path.join('lstm', 'results_binary', 'internal', 'round_' + str(n), 'predict_result_' + str(n) + '.csv')
#     read_path = os.path.join('lstm', 'results_binary', 'external', 'predict_result_' + str(n) + '.csv')
#     test_data = pd.read_csv(read_path)
#     predict_prob = test_data['stenosis_pred']
#     true_label = test_data['stenosis']
#     fpr, tpr, _ = roc_curve(true_label, predict_prob)
#     #p = auc(fpr, tpr)
#     # print(roc_auc)
#     predict_label = np.where(predict_prob > 0.5, 1, 0)
#     # p = f1_score(true_label, predict_label)
#     # p = average_precision_score(true_label, predict_prob, average='micro')
#     xlnet_precision, xlnet_recall, xlnet_thresholds = precision_recall_curve(true_label, predict_prob)
#     p = auc(xlnet_recall, xlnet_precision)
#     print(p)
#     ps.append(p)
# print(round(np.mean(ps), 2), round(np.std(ps), 2))

# Finally, 'macro' calculates the F1 separated by class but not using weights for the aggregation:
# 𝐹1𝑐𝑙𝑎𝑠𝑠1+𝐹1𝑐𝑙𝑎𝑠𝑠2+⋅⋅⋅+𝐹1𝑐𝑙𝑎𝑠𝑠𝑁
# which resuls in a bigger penalisation when your model does not perform well with the minority classes.