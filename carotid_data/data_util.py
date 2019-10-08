import pickle
import os
current_path = os.path.dirname(__file__)


def read_variable(val_name):
    open_path = os.path.join(current_path, val_name)
    with open(open_path, 'rb') as file_pi:
        val = pickle.load(file_pi)
        return val


def save_variable(val, val_name):
    save_path = os.path.join(current_path, val_name)
    with open(save_path, 'wb') as file_pi:
        pickle.dump(val, file_pi)