import pandas as pd
import numpy as np
import os

train_data = pd.read_csv(os.path.join('..', 'experiments', 'data', 'internal', 'round_0', 'training_0.csv'))

train_data['bi'] = np.where(train_data.iloc[:, 2:].sum(axis=1) > 0, 1, 0)
print('done')