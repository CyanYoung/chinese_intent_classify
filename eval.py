import pickle as pk

import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from classify import ind_labels, models

from util import flat_read, map_item


detail = False

path_test = 'data/test.csv'
path_ml_sent = 'feat/ml/sent_test.pkl'
path_nn_sent = 'feat/nn/sent_test.pkl'
path_label = 'feat/label_test.pkl'
texts = flat_read(path_test, 'text')
with open(path_ml_sent, 'rb') as f:
    ml_sents = pk.load(f)
with open(path_nn_sent, 'rb') as f:
    nn_sents = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)

class_num = len(ind_labels)

paths = {'svm': 'metric/svm.csv',
         'xgb': 'metric/xgb.csv',
         'dnn': 'metric/dnn.csv',
         'cnn': 'metric/cnn.csv',
         'rnn': 'metric/rnn.csv'}


def test(name, sents, labels):
    model = map_item(name, models)
    if name == 'svm' or name == 'xgb':
        probs = model.predict_proba(sents)
    else:
        probs = model.predict(sents)
    preds = np.argmax(probs, axis=1)
    precs = precision_score(labels, preds, average=None)
    recs = recall_score(labels, preds, average=None)
    with open(map_item(name, paths), 'w') as f:
        f.write('label,prec,rec' + '\n')
        for i in range(class_num):
            f.write('%s,%.2f,%.2f\n' % (ind_labels[i], precs[i], recs[i]))
    f1 = f1_score(labels, preds, average='weighted')
    print('\n%s f1: %.2f - acc: %.2f\n' % (name, f1, accuracy_score(labels, preds)))
    if detail:
        for text, label, pred in zip(texts, labels, preds):
            if label != pred:
                print('{}: {} -> {}'.format(text, ind_labels[label], ind_labels[pred]))


if __name__ == '__main__':
    test('svm', ml_sents, labels)
    test('xgb', ml_sents, labels)
    test('dnn', nn_sents, labels)
    test('cnn', nn_sents, labels)
    test('rnn', nn_sents, labels)
