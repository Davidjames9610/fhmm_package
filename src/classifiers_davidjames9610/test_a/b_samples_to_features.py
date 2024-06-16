
import src.ads_davidjames9610.ads as ads
import src.ads_davidjames9610.useful as useful
import numpy as np
import os
import os.path
from pathlib import Path
import matplotlib.pyplot as plt
from importlib import reload
from src.classifiers_davidjames9610.test_a.e_config import *
import src.misc_davidjames9610.fe_methods as fe
import src.misc_davidjames9610.proc_methods as pm
import random

# TODO expand this to include reverb
# annoying that I check for noise twice :/
def get_features(cv_output, fe_method, snr=None, signal_average_power=None):

    features = {}

    # train
    train_features = []
    train_labels = []
    n_folds = len(cv_output['train_data'])
    for i in range(n_folds):
        curr_features = [fe_method(sample) for sample in cv_output['train_data'][i]]
        curr_labels = cv_output['train_label'][i]
        train_features.append(curr_features)
        train_labels.append(curr_labels)

        # ng inf error handler
        # mask = [np.isneginf(array).any() for array in curr_features]
        # train_features.append([feat for feat in curr_features if not mask])
        # mask = np.invert(np.array(mask))
        # train_labels.append(cv_output['train_label'][i][mask])

    features['train_features'] = train_features
    features['train_label'] = train_labels

    # val
    val_features = []
    val_labels = []
    for i in range(n_folds):
        fold_samples = cv_output['val_data'][i]
        fold_features = []
        for sample in fold_samples:

            pass
        val_features.append([fe_method(sample) for sample in cv_output['val_data'][i]])
    features[fe_name]['val_features'] = val_features
    features[fe_name]['val_label'] = cv_output['val_label']

def get_average_power_for_samples(cv_output):
    # get average signal power
    curr_samples = cv_output['train_data'][0]
    random_indices = random.sample(range(1, len(curr_samples)), 20)
    samples = np.concatenate([curr_samples[indice] for indice in random_indices])
    ap = periodic_power(samples, 500, 250)
    ap = np.array(ap)
    ap = ap[ap > 0.05]
    # plt.plot(ap)
    # plt.show()
    # plt.close()
    return np.mean(ap)

def periodic_power(x, lx, p):
    buf = buffer(x, lx, p)
    average_pow = []
    for b in buf:
        average_pow.append(np.sum(np.square(b)) / lx)
    return average_pow

def buffer(x, n, p=0, opt=None):
    """
    Mimic MATLAB routine to generate buffer array

    MATLAB docs here: https://se.mathworks.com/help/signal/ref/buffer.html

    Parameters
    ----------
    x: ndarray
        Signal array
    n: int
        Number of data segments
    p: int
        Number of values to overlap
    opt: str
        Initial condition options. default sets the first `p` values to zero,
        while 'nodelay' begins filling the buffer immediately.

    Returns
    -------
    result : (n,n) ndarray
        Buffer array created from x
    """
    import numpy as np

    if opt not in [None, 'nodelay']:
        raise ValueError('{} not implemented'.format(opt))

    i = 0
    first_iter = True
    while i < len(x):
        if first_iter:
            if opt == 'nodelay':
                # No zeros at array start
                result = x[:n]
                i = n
            else:
                # Start with `p` zeros
                result = np.hstack([np.zeros(p), x[:n - p]])
                i = n - p
            # Make 2D array and pivot
            result = np.expand_dims(result, axis=0).T
            first_iter = False
            continue

        # Create next column, add `p` results from last col if given
        col = x[i:i + (n - p)]
        if p != 0:
            col = np.hstack([result[:, -1][-p:], col])
        i += n - p

        # Append zeros if last row and not length `n`
        if len(col) < n:
            col = np.hstack([col, np.zeros(n - len(col))])

        # Combine result with next row
        result = np.hstack([result, np.expand_dims(col, axis=0).T])

    return result.T
