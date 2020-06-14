import pandas as pd
import numpy as np
import os
import pickle


# read_path = os.path.join('xlnet', 'results', 'external', 'predict_result_0.pickle')
read_path = os.path.join('xlnet', 'results', 'internal', 'round_0', 'predict_result.pickle')
# cols = ['RIVA', 'LIVA', 'RPCA', 'LPCA']
cols = ['RIICA', 'RACA', 'RMCA', 'RPCA', 'RIVA', 'BA', 'LIICA', 'LACA', 'LMCA', 'LPCA', 'LIVA']
inx = []
with open(read_path, 'rb') as file:
    test_data = pickle.load(file)
    for col in cols:
        test_data[col+'_pred'] = np.where(test_data[col+'_pred'] > 0.5, 1, 0)
        i = list(test_data[test_data[col] != test_data[col+'_pred']].index.values)
        inx = inx + i
    test_data = test_data.iloc[inx]
test_data.to_csv('errors_all.csv')
print('done')