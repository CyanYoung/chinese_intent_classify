import pickle as pk

import numpy as np

from keras.models import load_model

from keras.preprocessing.sequence import pad_sequences

from preprocess import clean

from util import map_item


def ind2label(label_inds):
    ind_labels = dict()
    for label, ind in label_inds.items():
        ind_labels[ind] = label
    return ind_labels


seq_len = 30

path_bow = 'model/ml/bow.pkl'
path_svd = 'model/ml/svd.pkl'
path_svm = 'model/ml/svm.pkl'
path_xgb = 'model/ml/xgb.pkl'
with open(path_bow, 'rb') as f:
    bow = pk.load(f)
with open(path_svd, 'rb') as f:
    svd = pk.load(f)
with open(path_svm, 'rb') as f:
    svm = pk.load(f)
with open(path_xgb, 'rb') as f:
    xgb = pk.load(f)

path_word2ind = 'model/nn/word2ind.pkl'
path_label_ind = 'feat/label_ind.pkl'
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)

ind_labels = ind2label(label_inds)

paths = {'dnn': 'model/nn/dnn.h5',
         'cnn': 'model/nn/cnn.h5',
         'rnn': 'model/nn/rnn.h5'}

models = {'svm': svm,
          'xgb': xgb,
          'dnn': load_model(map_item('dnn', paths)),
          'cnn': load_model(map_item('cnn', paths)),
          'rnn': load_model(map_item('rnn', paths))}


def ml_predict(text, name):
    sent = bow.transform([text])
    sent = svd.transform(sent)
    model = map_item(name, models)
    return model.predict_proba(sent)[0]


def nn_predict(text, name):
    seq = word2ind.texts_to_sequences([text])[0]
    pad_seq = pad_sequences([seq], maxlen=seq_len)
    model = map_item(name, models)
    return model.predict(pad_seq)[0]


def predict(text, name):
    text = clean(text)
    if name == 'svm' or name == 'xgb':
        probs = ml_predict(text, name)
    else:
        probs = nn_predict(text, name)
    sort_probs = sorted(probs, reverse=True)
    sort_inds = np.argsort(-probs)
    sort_preds = [ind_labels[ind] for ind in sort_inds]
    formats = list()
    for pred, prob in zip(sort_preds, sort_probs):
        formats.append('{} {:.3f}'.format(pred, prob))
    return ', '.join(formats)


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('svm: %s' % predict(text, 'svm'))
        print('xgb: %s' % predict(text, 'xgb'))
        print('dnn: %s' % predict(text, 'dnn'))
        print('cnn: %s' % predict(text, 'cnn'))
        print('rnn: %s' % predict(text, 'rnn'))
