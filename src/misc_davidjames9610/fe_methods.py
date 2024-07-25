from random import random

import librosa
import torchaudio
from hmmlearn.hmm import GaussianHMM
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from spafe.utils import vis
import numpy as np
from torch.distributions import ComposeTransform
from IPython.display import Audio, display
from spafe.features.mfcc import *


class FeatureExtractorLogPower:
    """
    log power features
    used mainly for fhmm
    """

    def __init__(self, nfft):
        self.nfft = nfft

    def __str__(self):
        return f"lp"

    def __call__(self, sample):
        return np.log(np.square(np.abs(librosa.stft(sample, n_fft=self.nfft)).T))

class FeatureExtractorMfcc:
    """
    FE method mfcc
    """
    def __init__(self, nfft, num_ceps, fs):
        self.nfft = nfft
        self.num_ceps = num_ceps
        self.fs = fs

    def __str__(self):
        return 'mfcc'

    def __call__(self, sample):
        return zero_handling(mfcc(sig=sample, fs=self.fs, num_ceps=self.num_ceps, nfft=self.nfft))
