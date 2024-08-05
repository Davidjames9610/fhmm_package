import keras.optimizers
# combine lstm and deep nn with hmm


import numpy as np
import hmmlearn.hmm as hmmlearn
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Flatten
import tensorflow as tf
from keras.utils import to_categorical

def elog(x):
    res = np.log(x, where=(x != 0))
    res[np.where(x == 0)] = -(10.0 ** 8)
    return res

def getExpandedData(data):
    T = data.shape[0]

    data_0 = np.copy(data[0])
    data_T = np.copy(data[T - 1])

    for i in range(3):
        data = np.insert(data, 0, data_0, axis=0)
        data = np.insert(data, -1, data_T, axis=0)

    data_expanded = np.zeros((T, 7 * data.shape[1]))
    for t in range(3, T + 3):
        np.concatenate((data[t - 3], data[t - 2], data[t - 1], data[t],
                        data[t + 1], data[t + 2], data[t + 3]), out=data_expanded[t - 3])

    return data_expanded


def logSumExp(x, axis=None, keepdims=False):
    x_max = np.max(x, axis=axis, keepdims=keepdims)
    x_diff = x - x_max
    sumexp = np.exp(x_diff).sum(axis=axis, keepdims=keepdims)
    return x_max + np.log(sumexp)


class LogProbLayer(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.log(inputs)

class KerasHMM:
    def __init__(self, n_components=8, n_mix=2, lstm=True, verbose=False):
        self.n_mix = n_mix
        self.n_components = n_components
        self.hmm = hmmlearn.GMMHMM(n_components=n_components, n_mix=n_mix) # , n_mix=n_mix)  # todo update to GMM
        self.nn = None
        self.lstm = lstm
        self.verbose = verbose

    # features should not be concatenated
    def fit(self, features):
        self.train_hmm(features)
        # sequences = self.hmm.predict(features)
        posterior_probs = self.hmm.predict_proba(features)
        self.train_nn(features, posterior_probs)

    def train_hmm(self, features):
        # lengths = []
        # for n, i in enumerate(features):
        #     lengths.append(len(i))
        # self.hmm.fit(np.concatenate(features), lengths)
        self.hmm.fit(features)

    def train_nn(self, features, sequences):

        # one_hot_encoded = to_categorical(sequences, num_classes=self.n_components)
        dim = features.shape[1]

        model = Sequential()
        if self.lstm:
            model.add(LSTM(150, activation='relu', input_shape=(dim, 1), return_sequences=True))
            model.add(LSTM(100, activation='relu', input_shape=(dim, 1), return_sequences=True))
            model.add(LSTM(25, activation='relu'))
            model.add(Dense(self.n_components, activation='softmax'))
            # model.add(Activation('softmax'))
            # model.add(LogProbLayer())
            model.compile(loss='categorical_crossentropy', optimizer='adam')
        else:
            model.add(Dense(100, activation='relu', input_shape=(dim, 1)))
            model.add(Flatten())
            model.add(Dense(50, activation='relu'))
            # model.add(Dense(25, activation='relu'))
            model.add(Dense(self.n_components, activation='softmax'))
            optimizer = keras.optimizers.Adam(learning_rate=0.0001)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer)

        model.fit(features, sequences, verbose=self.verbose)

        self.nn = model

    def mlp_predict(self, o):
        return np.log(self.nn.predict(o, verbose=self.verbose) + 1e-8)

    def viterbi_mlp(self, o):

        hmm = self.hmm
        pi = hmm.startprob_
        a = hmm.transmat_

        T = o.shape[0]
        J = len(pi)

        s_hat = np.zeros(T, dtype=int)

        log_delta = np.zeros((T, J))

        psi = np.zeros((T, J))

        log_delta[0] = elog(pi)

        mlp_ll = self.mlp_predict(o)

        log_delta[0] += np.array([mlp_ll[0][j] for j in range(J)])

        log_A = elog(a)

        for t in range(1, T):
            for j in range(J):
                temp = np.zeros(J)
                for i in range(J):
                    temp[i] = log_delta[t - 1, i] + log_A[i, j] + mlp_ll[t][j]
                log_delta[t, j] = np.max(temp)
                psi[t, j] = np.argmax(log_delta[t - 1] + log_A[:, j])

        s_hat[T - 1] = np.argmax(log_delta[T - 1])

        for t in reversed(range(T - 1)):
            s_hat[t] = psi[t + 1, s_hat[t + 1]]

        return s_hat

    def forward_mlp(self, o):

        hmm = self.hmm

        pi = hmm.startprob_
        a = hmm.transmat_

        T = o.shape[0]
        J = len(pi)

        log_alpha = np.zeros((T, J))
        log_alpha[0] = elog(pi)

        mlp_ll = self.mlp_predict(o)    # t x f

        log_alpha[0] += np.array([mlp_ll[0][j] for j in range(J)])

        for t in range(1, T):
            for j in range(J):
                mlp_ll_t = mlp_ll[t][j]
                log_alpha[t, j] = mlp_ll_t + logSumExp(elog(a[:, j].T) + log_alpha[t - 1])

        return log_alpha

    # NB for getting results
    def score(self, data):
        T = data.shape[0]
        log_alpha_t = self.forward_mlp(data)[T - 1]
        ll = logSumExp(log_alpha_t)
        return ll
