import logging
import src.fhmm_davidjames9610.fhmm_utils as fhmm_utils
import numpy as np 
from hmmlearn.hmm import GaussianHMM
import pytest

def create_diag_gaussian_hmm(n_components, means, covariances):
    model = GaussianHMM(n_components=n_components, covariance_type="diag")
    model.means_ = means
    model.covars_ = covariances
    model.n_features = model.means_.shape[1]
    return model

def test_get_combined_mean_covariance_and_state_dict():

    model_a = create_diag_gaussian_hmm(
        n_components=3,
        means=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        covariances=np.array([[1.0, 0.5], [0.8, 0.4], [1.0, 0.5]])
    )

    model_b = create_diag_gaussian_hmm(
        n_components=2,
        means=np.array([[3.0, 0.0], [2.0, 1.0]]),
        covariances=np.array([[1.0, 0.5], [0.8, 0.4]])
    )

    combined_mean, combined_covars, state_dict = fhmm_utils.get_combined_mean_covariance_and_state_dict(model_a, model_b)

    assert combined_mean.shape == (6,2)
    assert combined_covars.shape == (6,2,2)
    assert len(state_dict) == 6

def generate_gaussian_noise(num_samples, power_a, power_b, power_c):
    # Define the parameters of the HMM
    startprob = np.array([0.3, 0.3, 0.4])
    transmat = np.array([[0.999, 0.0005, 0.0005], [0.0005, 0.999, 0.0005], [0.0005,0.0005, 0.999]])  # Transition matrix
    means = np.array([[0.0], [0.0], [0.0]])  # Mean values for each state
    covars = np.array([[[power_a]],[[power_b]],[[power_c]]])  # Covariance matrices for each state

    # Create a two-state HMM
    model = GaussianHMM(n_components=len(startprob), covariance_type="full", n_iter=100)
    model.n_features = 1
    model.startprob_ = startprob
    model.transmat_ = transmat
    model.means_ = means
    model.covars_ = covars

    # Generate samples from the HMM
    noise, states = model.sample(num_samples)

    return noise.flatten(), states

def get_noise_avg_watts(data, snr):
    x_watts = data ** 2
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    target_snr_db = snr
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    return noise_avg_watts

def normalize(audio):
    audio = np.where(audio == 0, audio + 0.001, audio)
    std = (np.round(np.std(audio) * 1000) / 1000) * 10  # 97%
    mean = np.mean(audio)
    new_audio = (audio - mean) / std
    if np.mean(new_audio) > 0.01:
        raise logging.warning('mean is large')
    return new_audio

