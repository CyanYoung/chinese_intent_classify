import pickle as pk

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import TruncatedSVD

from vectorzie import label2ind

from util import flat_read


min_freq = 5

path_bow = 'model/ml/bow.pkl'
path_svd = 'model/ml/svd.pkl'
path_label_ind = 'feat/label_ind.pkl'


def sent2feat(sents, path_bow, path_svd, mode):
    if mode == 'train':
        bow = CountVectorizer(token_pattern='\w', min_df=min_freq)
        bow.fit(sents)
        with open(path_bow, 'wb') as f:
            pk.dump(bow, f)
    else:
        with open(path_bow, 'rb') as f:
            bow = pk.load(f)
    bow_sents = bow.transform(sents)
    if mode == 'train':
        svd = TruncatedSVD(n_components=200, n_iter=10)
        svd.fit(bow_sents)
        with open(path_svd, 'wb') as f:
            pk.dump(svd, f)
    else:
        with open(path_svd, 'rb') as f:
            svd = pk.load(f)
    return svd.transform(bow_sents)


def featurize(path_data, path_sent, path_label, mode):
    sents = flat_read(path_data, 'text')
    labels = flat_read(path_data, 'label')
    sent_feats = sent2feat(sents, path_bow, path_svd, mode)
    if mode == 'train':
        label2ind(labels, path_label_ind)
    with open(path_label_ind, 'rb') as f:
        label_inds = pk.load(f)
    inds = list()
    for label in labels:
        inds.append(label_inds[label])
    inds = np.array(inds)
    with open(path_sent, 'wb') as f:
        pk.dump(sent_feats, f)
    with open(path_label, 'wb') as f:
        pk.dump(inds, f)


if __name__ == '__main__':
    path_data = 'data/train.csv'
    path_sent = 'feat/ml/sent_train.pkl'
    path_label = 'feat/label_train.pkl'
    featurize(path_data, path_sent, path_label, 'train')
    path_data = 'data/test.csv'
    path_sent = 'feat/ml/sent_test.pkl'
    path_label = 'feat/label_test.pkl'
    featurize(path_data, path_sent, path_label, 'test')
