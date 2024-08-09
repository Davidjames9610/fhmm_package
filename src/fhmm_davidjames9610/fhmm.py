from hmmlearn.hmm import GaussianHMM
import librosa
import numpy as np
from src.fhmm_davidjames9610.fhmm_utils import get_combined_mean_covariance_and_state_dict, kronecker_list, get_soft_mask

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
        self.state_dict = None

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
        self.combine_given_hmms(self.hmm_a, self.hmm_b)

    def fit_given_signal_hmm(self, hmm_a, features_b):
        # hmm
        self.hmm_a = hmm_a
        self.n_components_a = hmm_a.n_components
        self.hmm_b.fit(features_b)
        self.combine_given_hmms(self.hmm_a, self.hmm_b)

    def combine_given_hmms(self, hmm_a, hmm_b):

        self.hmm_a = hmm_a
        self.n_components_a = hmm_a.n_components
        self.hmm_b = hmm_b
        self.n_components_b = hmm_b.n_components

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
        hmm_combined.n_features = self.hmm_a.n_features
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

        log_prob, states_decoded = self.hmm_combined.decode(X)

        # split into separate:

        ss01 = []
        ss02 = []
        for x in range(len(states_decoded)):
            temp = self.state_dict[states_decoded[x]]
            ss01.append(temp[0])
            ss02.append(temp[1])

        return log_prob, [ss01, ss02]

    def score(self, X):
        log_prob, _ = self.decode(X)
        return log_prob

    # split state sequence into two
    def seperate_features(self, X, mask_type='soft'):
        """
        Decode mixed feature into clean audio
        
        Parameters
        ----------
        X: np.array
            log-power features of signals combined
        
        Returns
        -------
        feat_a: array, shape (n_samples, )
            features for hmm_a
        
        feat_b: array, shape (n_samples, )
            features for hmm_b
        """
        _, [ss_a, ss_b] = self.decode(X)

        # if mask_type == 'soft':
        mask_a, mask_b = get_soft_mask(X, self.hmm_a.means_, ss_a, self.hmm_b.means_, ss_b)

        feat_a = ((np.exp(X)) * mask_a)
        feat_b = ((np.exp(X)) * mask_b)

        return feat_a, feat_b

    def get_clean_audio(self, X, nfft, mask_type ='soft'):
        """
        Decode mixed feature into clean audio
        
        Parameters
        ----------
        X: np.array
            log-power features of signals combined

        nfft: int
            used to convert back to time domain
        
        Returns
        -------
        sig_a : array, shape (n_samples, )
            clean audio in time domain for hmm_a
        
        sig_b : array, shape (n_samples, )
            clean audio in time domain for hmm_b.

        uses 'Approximate magnitude spectrogram inversion using the "fast" Griffin-Lim algorithm' from librosa documentation
        """

        feat_a, feat_b = self.seperate_features(X, mask_type)

        feat_a_stft = np.sqrt(feat_a)
        feat_b_stft = np.sqrt(feat_b)

        sig_a = librosa.griffinlim(feat_a_stft.T, n_fft=nfft)
        sig_b = librosa.griffinlim(feat_b_stft.T, n_fft=nfft)

        return sig_a, sig_b

if __name__ == '__main__':

    # simple example, after this get un_mix working and include audio processing 
    # features_a, features_b, features_combined
    # my_fhmm = FHMM(n_components_a, n_components_b)
    # my_fhmm.fit(features_a, features_b)
    # log_prob, state_sequence_a, state_sequence_b = my_fhmm.decode(features_combined)

    print('hi')

