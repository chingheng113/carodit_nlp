import pandas as pd
import pickle
import os

read_path = os.path.join('results', 'internal', 'round_0', 'predict_result.pickle')
with open(read_path, 'rb') as file:
    predict =pickle.load(file)
    print()

read_path = os.path.join('results', 'internal', 'round_0', 'elapse_time.pickle')
with open(read_path, 'rb') as file:
    elapse_time =pickle.load(file)
    print(elapse_time)