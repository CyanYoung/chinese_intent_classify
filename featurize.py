import pickle as pk

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from vectorzie import label2ind

from util import flat_read


min_freq = 5

path_bow = 'model/ml/bow.pkl'
path_tfidf = 'model/ml/tfidf.pkl'
path_label_ind = 'feat/label_ind.pkl'


def bow(sents, path_bow, mode):
    if mode == 'train':
        model = CountVectorizer(token_pattern='\w', min_df=min_freq)
        model.fit(sents)
        with open(path_bow, 'wb') as f:
            pk.dump(model, f)
    else:
        with open(path_bow, 'rb') as f:
            model = pk.load(f)
    return model.transform(sents)


def tfidf(bow_sents, path_tfidf, mode):
    if mode == 'train':
        model = TfidfTransformer()
        model.fit(bow_sents)
        with open(path_tfidf, 'wb') as f:
            pk.dump(model, f)
    else:
        with open(path_tfidf, 'rb') as f:
            model = pk.load(f)
    return model.transform(bow_sents)


def featurize(paths, mode):
    sents = flat_read(paths['data'], 'text')
    labels = flat_read(paths['data'], 'label')
    bow_sents = bow(sents, path_bow, mode)
    tfidf_sents = tfidf(bow_sents, path_tfidf, mode)
    if mode == 'train':
        label2ind(labels, path_label_ind)
    with open(path_label_ind, 'rb') as f:
        label_inds = pk.load(f)
    inds = list()
    for label in labels:
        inds.append(label_inds[label])
    inds = np.array(inds)
    with open(paths['bow_sent'], 'wb') as f:
        pk.dump(bow_sents, f)
    with open(paths['tfidf_sent'], 'wb') as f:
        pk.dump(tfidf_sents, f)
    with open(paths['label'], 'wb') as f:
        pk.dump(inds, f)


if __name__ == '__main__':
    paths = dict()
    prefix = 'feat/ml/'
    paths['data'] = 'data/train.csv'
    paths['bow_sent'] = prefix + 'bow_sent_train.pkl'
    paths['tfidf_sent'] = prefix + 'tfidf_sent_train.pkl'
    paths['label'] = 'feat/label_train.pkl'
    featurize(paths, 'train')
    paths['data'] = 'data/test.csv'
    paths['bow_sent'] = prefix + 'bow_sent_test.pkl'
    paths['tfidf_sent'] = prefix + 'tfidf_sent_test.pkl'
    paths['label'] = 'feat/label_test.pkl'
    featurize(paths, 'test')
