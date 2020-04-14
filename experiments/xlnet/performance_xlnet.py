import pandas as pd
import pickle
import os


in_cols = ['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA', 'LCCA', 'LEICA', 'LIICA',
           'LACA', 'LMCA', 'LPCA', 'LEVA', 'LIVA']

for n in range(10):
    read_path = os.path.join('results', 'internal', 'round_'+str(n), 'predict_result.pickle')
    with open(read_path, 'rb') as file:
        test_data =pickle.load(file)
        true_label = test_data[in_cols]
        pred_label = test_data[[s+'_pred' for s in in_cols]]


        print()





read_path = os.path.join('results', 'internal', 'round_0', 'elapse_time.pickle')
with open(read_path, 'rb') as file:
    elapse_time =pickle.load(file)
    print(elapse_time)