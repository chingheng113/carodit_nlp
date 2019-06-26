import data_util
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os
current_path = os.path.dirname(__file__)

test_data = data_util.read_variable('testing_data.pickle')
t2s_test_pad = test_data[0]
Y_test = test_data[1]
model = load_model('sequential_1.h5')
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
y_pred_p = model.predict(t2s_test_pad)
print(y_pred_p)
for i in range(0, 17):
    p_label = Y_test[:, i]
    p_label_b = (p_label > 0.5).astype(int)
    t_label = Y_test[:, i]
    fpr, tpr, thresholds = roc_curve(t_label, p_label)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    # plt.plot(fpr, tpr, label='(AUC = %0.3f )' % roc_auc, lw=1, alpha=.8)
    print(classification_report(t_label, p_label_b))
    print(confusion_matrix(t_label, p_label_b))
print('done')