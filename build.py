import pickle as pk

from sklearn.svm import SVC

from xgboost.sklearn import XGBClassifier as XGBC

from keras.models import Model
from keras.layers import Input, Embedding
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model

from nn_arch import dnn, cnn, rnn

from util import map_item


batch_size = 32

path_bow_sent = 'feat/ml/bow_sent_train.pkl'
path_tfidf_sent = 'feat/ml/tfidf_sent_train.pkl'
with open(path_bow_sent, 'rb') as f:
    bow_sents = pk.load(f)
with open(path_tfidf_sent, 'rb') as f:
    tfidf_sents = pk.load(f)

path_embed = 'feat/nn/embed.pkl'
path_sent = 'feat/nn/sent_train.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_sent, 'rb') as f:
    sents = pk.load(f)

path_label_ind = 'feat/label_ind.pkl'
path_label = 'feat/label_train.pkl'
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)

class_num = len(label_inds)

feats = {'bow': bow_sents,
         'tfidf': tfidf_sents}

funcs = {'dnn': dnn,
         'cnn': cnn,
         'rnn': rnn}

paths = {'svm': 'model/ml/svm.pkl',
         'xgb': 'model/ml/xgb.pkl',
         'dnn': 'model/nn/dnn.h5',
         'cnn': 'model/nn/cnn.h5',
         'rnn': 'model/nn/rnn.h5',
         'dnn_plot': 'model/nn/plot/dnn.png',
         'cnn_plot': 'model/nn/plot/cnn.png',
         'rnn_plot': 'model/nn/plot/rnn.png'}


def svm_fit(feat, labels):
    sents = map_item(feat, feats)
    model = SVC(C=10.0, kernel='linear', max_iter=1000, probability=True,
                class_weight='balanced', verbose=True)
    model.fit(sents, labels)
    with open(map_item('svm', paths), 'wb') as f:
        pk.dump(model, f)


def xgb_fit(feat, labels):
    sents = map_item(feat, feats)
    model = XGBC(max_depth=5, learning_rate=0.1, objective='binary:logistic',
                 n_estimators=100, booster='gbtree')
    model.fit(sents, labels)
    with open(map_item('xgb', paths), 'wb') as f:
        pk.dump(model, f)


def nn_compile(name, embed_mat, seq_len, class_num):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len,
                      weights=[embed_mat], input_length=seq_len, trainable=True)
    input = Input(shape=(seq_len,))
    embed_input = embed(input)
    func = map_item(name, funcs)
    output = func(embed_input, class_num)
    model = Model(input, output)
    model.summary()
    plot_model(model, map_item(name + '_plot', paths), show_shapes=True)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model


def nn_fit(name, epoch, embed_mat, class_num, sents, labels):
    seq_len = len(sents[0])
    model = nn_compile(name, embed_mat, seq_len, class_num)
    check_point = ModelCheckpoint(map_item(name, paths), monitor='val_loss', verbose=True, save_best_only=True)
    model.fit(sents, labels, batch_size=batch_size, epochs=epoch,
              verbose=True, callbacks=[check_point], validation_split=0.2)


if __name__ == '__main__':
    svm_fit('bow', labels)
    xgb_fit('bow', labels)
    nn_fit('dnn', 10, embed_mat, class_num, sents, labels)
    nn_fit('cnn', 10, embed_mat, class_num, sents, labels)
    nn_fit('rnn', 10, embed_mat, class_num, sents, labels)
