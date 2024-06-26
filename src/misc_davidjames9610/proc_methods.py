
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

def file_to_audio(file_dr, sample_rate):
    """
    file to audio using torchaudio transforms to normalise audio too

    Args:
        file_dr: location of file
        sample_rate: desired sample_rate

    Returns:
        audio: n x 1 numpy array
    """
    effects = [
        ['remix', '1'],  # convert to mono
        ['rate', str(sample_rate)],  # resample
        ['gain', '-n']  # normalises to 0dB
    ]
    audio, sr = torchaudio.sox_effects.apply_effects_file(file_dr, effects, normalize=True)

    return audio.numpy().flatten()

def get_audio_sr(file):
    y, sr = librosa.load(file)
    print('natural sr is: ', sr)
    return sr

