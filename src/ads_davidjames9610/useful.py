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


def file_to_audio(file_dr, sample_rate, vad=False, trigger_level=7):
    effects = [
        ['remix', '1'],  # convert to mono
        ['rate', str(sample_rate)],  # resample
        ['gain', '-n']  # normalises to 0dB
    ]
    audio, sr = torchaudio.sox_effects.apply_effects_file(file_dr, effects, normalize=True)

    if vad:
        compose_transform = torchaudio.transforms.Vad(
            sample_rate=sample_rate, trigger_level=trigger_level)

        audio = compose_transform(audio)

    return audio.numpy().flatten(), sr


class RandomClip:
    def __init__(self, sample_rate, clip_length):
        self.clip_length = clip_length
        self.vad = torchaudio.transforms.Vad(
            sample_rate=sample_rate, trigger_level=7.0)

    def __call__(self, audio_data):
        audio_length = audio_data.shape[0]
        if audio_length > self.clip_length:
            offset = random.randint(0, audio_length - self.clip_length)
            audio_data = audio_data[offset:(offset + self.clip_length)]

        return self.vad(audio_data)  # remove silences at the begining/end

class SampleHolder:
    def __init__(self, samples, sample_labels, features=None, feature_labels=None):
        if feature_labels is None:
            feature_labels = []
        if features is None:
            features = []
        self.samples = samples  # n x t x 1
        self.sample_labels = sample_labels  # n x 1
        self.features = features  # n x ft x n_features
        self.feature_labels = feature_labels  # n x ft
        self.log_prob = None
        self.y_pred = None

    def update_feature_labels(self):
        if len(self.features) > 0:
            self.feature_labels = []
            for i in range(len(self.features)):
                self.feature_labels.append(np.ones(len(self.features[i])) * self.sample_labels[i])

    def update_decode(self, output):
        self.y_pred = output.y_pred
        self.log_prob = output.log_prob

    def remove_index(self, some_index):
        if len(self.feature_labels) > 0:
            np.delete(self.feature_labels, some_index)
        if len(self.features) > 0:
            np.delete(self.features, some_index)
        if len(self.samples) > 0:
            np.delete(self.samples, some_index)
        if len(self.sample_labels) > 0:
            np.delete(self.sample_labels, some_index)


def get_log_power_feature(sample, nfft):
    return np.log(np.square(np.abs(librosa.stft(sample, n_fft=nfft)).T))

def get_average_cm(conf_matrices):

    # Initialize a variable to store the sum of confusion matrices
    sum_conf_matrix = np.zeros(conf_matrices[0].shape)

    # Sum up all confusion matrices
    for conf_matrix in conf_matrices:
        sum_conf_matrix += conf_matrix

    # Calculate the average confusion matrix
    avg_conf_matrix = sum_conf_matrix / len(conf_matrices)

    return avg_conf_matrix

def play_audio(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")