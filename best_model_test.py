import data_util
from keras.models import load_model
import os
current_path = os.path.dirname(__file__)

test_data = data_util.read_variable('testing_data.pickle')
t2s_test_pad = test_data[0]
Y_test = test_data[1]
model = load_model('sequential_1.h5')
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
y_pred_p = model.predict_classes(t2s_test_pad)
print(y_pred_p)
print('done')