from hmmlearn.hmm import GaussianHMM
import numpy as np
from src.fhmm_davidjames9610.fhmm_utils import get_combined_mean_covariance_and_state_dict, kronecker_list


# use combined not comb

class FHMM:

    def __init__(self, n_components_a, n_components_b):
        """
        init function
        
        Parameters:
        n_components_a: int
            number of components for hmm_a
        n_components_b: int
            number of components for hmm_b
        """
        self.n_components_a = n_components_a
        self.n_components_b = n_components_b
        self.hmm_a = GaussianHMM(n_components_a)
        self.hmm_b = GaussianHMM(n_components_b)
        self.hmm_combined = None
        self.init = True


    def fit(self, features_a, features_b):
        """
        fit hmms function.
        
        Parameters
        ----------
        features_a: array-like, shape (n_samples, n_features)
            log-power features for signal_a
        features_b: array-like, shape (n_samples, n_features)
            log-power features for signal_b
        """
        # hmm 
        self.hmm_a.fit(features_a)
        self.hmm_b.fit(features_b)

        # combine mean and covariance
        combined_mean, combined_covariance, state_dict = get_combined_mean_covariance_and_state_dict(self.hmm_a, self.hmm_b)

        # combine pi and A
        pi_combined = kronecker_list([self.hmm_a.startprob_, self.hmm_b.startprob_])
        pi_combined = pi_combined + 1e-10
        pi_combined /= pi_combined.sum()

        a_combined = kronecker_list([self.hmm_a.transmat_, self.hmm_b.transmat_]) + 1e-10
        a_combined /= a_combined.sum(axis=1)

        # create HMM
        hmm_combined = GaussianHMM(self.n_components_a * self.n_components_b, covariance_type='diag')
        hmm_combined.transmat_, hmm_combined.startprob_, hmm_combined.means_, hmm_combined.covars_ = a_combined, pi_combined, combined_mean, combined_covariance

        # combine things
        self.hmm_combined = hmm_combined
        self.state_dict = state_dict

    def decode(self, X):
        """
        Decode features combined into log_probability and 
        distinct state sequence for each hmm.
        
        Parameters
        ----------
        X: np.array
            log-power features of signals combined
        
        Returns
        -------
        state_sequence_a : array, shape (n_samples, )
            Labels for each sample from ``X`` obtained via a given
                decoder ``algorithm``.

        state_sequence_b : array, shape (n_samples, )
        np.arrays decoded state sequence,
        log probability of state sequence
        """
        pass


if __name__ == '__main__':

    # simple example, after this get un_mix working and include audio processing 
    # features_a, features_b, features_combined
    # my_fhmm = FHMM(n_components_a, n_components_b)
    # my_fhmm.fit(features_a, features_b)
    # log_prob, state_sequence_a, state_sequence_b = my_fhmm.decode(features_combined)

    print('hi')

