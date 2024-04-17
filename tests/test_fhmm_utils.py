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

    



    
