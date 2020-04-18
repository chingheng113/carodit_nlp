import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

labels = ['RIICA', 'RACA', 'RMCA', 'RPCA', 'RIVA', 'BA', 'LIICA', 'LACA', 'LMCA', 'LPCA', 'LIVA']


# Note that in binary classification, recall of the positive class is also known as “sensitivity”; recall of the negative class is “specificity”.


result = pd.read_csv('external.csv')
for label in labels:
    true_label = result[label]
    pred_label = result[label+'1']
    print(label)
    print(classification_report(true_label, pred_label))
    cm = confusion_matrix(true_label, pred_label)
    sensitivity = float(cm[0][0]) / np.sum(cm[0])
    specificity = float(cm[1][1]) / np.sum(cm[1])
    print(sensitivity)
    print(specificity)


# Finally, 'macro' calculates the F1 separated by class but not using weights for the aggregation:
#
# 𝐹1𝑐𝑙𝑎𝑠𝑠1+𝐹1𝑐𝑙𝑎𝑠𝑠2+⋅⋅⋅+𝐹1𝑐𝑙𝑎𝑠𝑠𝑁
#
# which resuls in a bigger penalisation when your model does not perform well with the minority classes.