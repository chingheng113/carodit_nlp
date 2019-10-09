import os
import numpy as np

labels = ['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA', 'LCCA', 'LEICA', 'LIICA',
          'LACA', 'LMCA', 'LPCA', 'LEVA', 'LIVA']
for label in labels:
    f_arr = []
    for inx in range(10):
        file_name = os.path.join('..', 'r_results', 'round_'+str(inx), label+'_all.txt')
        with open(file_name) as f:
            content = f.readlines()
            for s in content:
                if 'Sensitivity' in s:
                    sensitivity = float(s.split(':')[1])
                if 'Specificity' in s:
                    specificity = float(s.split(':')[1])
            f_score = 2*((sensitivity*specificity)/(sensitivity+specificity))
            f_score = round(f_score, 3)
            f_arr.append(f_score)
    f_mean = round(np.mean(f_arr, axis=0), 3)
    f_std = round(np.std(f_arr, axis=0), 3)

    # print(label)
    print(str(f_mean)+'_'+str(f_std))

