import numpy as np
import matplotlib.pyplot as plt
from spafe.features import mfcc
import librosa.util as util
import librosa
from spafe.utils import vis

def get_feature(sample, nfft):
    return np.log(np.abs(librosa.stft(sample, n_fft=nfft)).T)

def create_data():
    # load files
    file_a = librosa.example('humpback')
    y_a, sr = librosa.load(file_a)
    y_a = librosa.util.normalize(y_a)

    file_b = librosa.example('robin')
    y_b, sr = librosa.load(file_b)
    y_b = librosa.util.normalize(y_b)

    # split whale to same length as robin 
    oft = 80000
    y_a = y_a[oft:len(y_b) + oft]

    # combine sounds in time domain
    y_c = y_a + y_b

    nfft = 128
    features_a = get_feature(y_a, nfft=nfft)
    features_b = get_feature(y_b, nfft=nfft)
    feature_combined = get_feature(y_c, nfft=nfft)

    np.savetxt('./tests/test_data/whale_features.csv', features_a, delimiter=',', fmt='%f')
    np.savetxt('./tests/test_data/robin_features.csv', features_b, delimiter=',', fmt='%f')
    np.savetxt('./tests/test_data/combined_features.csv', feature_combined, delimiter=',', fmt='%f')


