import cnnmodel
from sklearn.metrics import auc as auc3
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
import numpy as np
import keras
import random
import sklearn.preprocessing as prep

def standard_scale(X_train):

    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    return X_train

def calculate_performace(test_num, pred_y, labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    Acc = float(tp + tn) / test_num

    if tp == 0 and fp == 0:
        precision = 0
        MCC = 0
        F1_score = 0
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
    else:
        precision = float(tp) / (tp + fp)
        sensitivity = float(tp) / (tp + fn)
        specificity = float(tn) / (tn + fp)
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        F1_score = float(2 * tp) / ((2 * tp) + fp + fn)

    return Acc, precision, sensitivity, specificity, MCC, F1_score

roc=[]
pr=[]
ACC=[]
spe=[]
Precision=[]
predict=[]
Recall=[]
F1=[]
mcc=[]
k=10
pfeature=np.loadtxt('../best_select/Pfeature.txt')
nfeature=np.loadtxt('../best_select/Nfeature.txt')
pfeature=standard_scale(pfeature)
nfeature=standard_scale(nfeature)
plabel=np.loadtxt('../best_select/Plabel.txt')
nlabel=np.loadtxt('../best_select/Nlabel.txt')
pindex=np.arange(len(pfeature))%k
nindex=np.arange(len(nfeature))%k
np.random.shuffle(pindex)
np.random.shuffle(nindex)
for fold in range(k):
    Train=np.concatenate([pfeature[pindex != fold],nfeature[nindex != fold]],axis=0)
    train=np.expand_dims(Train,2)
    train_labels=np.concatenate([plabel[pindex != fold],nlabel[nindex != fold]],axis=0)
    Test = np.concatenate([pfeature[pindex == fold], nfeature[nindex == fold]], axis=0)
    test = np.expand_dims(Test,2)
    test_label=np.concatenate([plabel[pindex == fold],nlabel[nindex == fold]],axis=0)
    model = keras.models.load_model('RunCnn.model')

    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='accuracy',
            patience=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath='model.h5',
            monitor='val_loss',
            save_best_only=True,
        )
    ]
    history=model.fit(train, train_labels,
              batch_size=64,
              epochs=80,
              callbacks=callbacks_list,
                      )
    np.set_printoptions(precision=3)
    predict_y2 = model.predict_proba(test)
    c = roc_auc_score(test_label, predict_y2)
    precision, recall, pr_thresholds = precision_recall_curve(test_label, predict_y2)
    d = auc3(recall, precision)

    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + precision[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]
    predicted_score = np.zeros(len(test_label))
    predicted_score[predict_y2[:, 0] > threshold] = 1
    confusion_matri = confusion_matrix(y_true=test_label, y_pred=predicted_score)
    Acc, precision, sensitivity, specificity, MCC, F1_score = calculate_performace(len(test_label), predicted_score, test_label)
    roc.append(c)
    pr.append(d)
    Precision.append(precision)
    Recall.append(sensitivity)
    ACC.append(Acc)
    mcc.append(MCC)
    spe.append(specificity)
    F1.append(F1_score)
    predict.append(predicted_score)
print("roc:", roc)
print("pr：", pr)
print("auroc = {:.5f}".format(np.mean(roc)))
print("auprc = {:.5f}".format(np.mean(pr)))
print("Acc = {:.5f}".format(np.mean(ACC)))
print("Pre = {:.5f}".format(np.mean(Precision)))
print("F = {:.5f}".format(np.mean(F1)))
print("Re = {:.5f}".format(np.mean(Recall)))
print("Spe = {:.5f}".format(np.mean(spe)))
print("MC = {:.5f}".format(np.mean(mcc)))


