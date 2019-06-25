import pandas as pd
import numpy as np
import pickle
import os
current_path = os.path.dirname(__file__)
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt


with open('predict_y.pickle', 'rb') as file:
    result =pickle.load(file)
    predict_y_p = result[:,0:17]
    true_y = result[:,17:].astype(int)
    # true_y = (true_y == 1)

    for i in range(0, 17):
        p_label = predict_y_p[:, i]
        p_label_b = (p_label > 0.5).astype(int)
        t_label = true_y[:, i]
        fpr, tpr, thresholds = roc_curve(t_label, p_label)
        roc_auc = auc(fpr, tpr)
        print(roc_auc)
        # plt.plot(fpr, tpr, label='(AUC = %0.3f )' % roc_auc, lw=1, alpha=.8)
        print(classification_report(t_label, p_label_b))
        print(confusion_matrix(t_label, p_label_b))
    print('done')