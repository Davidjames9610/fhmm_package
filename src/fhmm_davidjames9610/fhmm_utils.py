
from hmmlearn.hmm import GaussianHMM
import numpy as np

def get_combined_mean_covariance_and_state_dict(hmm_a: GaussianHMM, hmm_b: GaussianHMM):
        """
        Take two HMMs and generate a combined mean, covariance and a state dictionary used for decoding
        
        Parameters
        ----------
        hmm_a: GaussianHMM
            first GaussianHMM
        hmm_b: GaussianHMM
            second GaussianHMM 
        
        Returns
        -------
        means_combined : dict, shape (n_components_a * n_components_b, n_features)
        covars_combined : dict, shape (n_components_a * n_components_b, n_features, n_features)
        state_dictionary : array, shape (n_components_a * n_components_b, n_hmms
        """
        n_components_a = hmm_a.n_components
        n_components_b = hmm_b.n_components
        means_a = hmm_a.means_
        means_b = hmm_b.means_
        covars_a = hmm_a.covars_
        covars_b = hmm_b.covars_
        states_dict = []
        means_combined = []
        covars_combined = []
        for j in range(n_components_a):
            for k in range(n_components_b):
                mean_j = means_a[j]
                mean_k = means_b[k]
                means_combined.append(np.maximum(mean_j, mean_k))
                m_mask = (mean_j > mean_k)
                covar_j = np.diag(covars_a[j])
                covar_k = np.diag(covars_b[k])
                covars_combined.append(np.where(m_mask, covar_j, covar_k))
                states_dict.append([j, k])
        return np.array(means_combined), np.array(covars_combined), states_dict

def kronecker_list(list_A):
    '''
    Input: list_pi: List of PI's of individual learnt HMMs
    Output: Combined Pi for the FHMM
    '''
    result=list_A[0]
    for i in range(len(list_A)-1):
        result=np.kron(result,list_A[i+1])
    return result

def get_soft_mask(features, a_mean, a_ss, b_mean, b_ss, alpha=1):
    '''
        Parameters
        ----------
        features: array
            mixed features array
        a_mean: array
            mean of hmm_a
        a_ss: array
            state sequence for hmm_a
        
        Returns
        -------
        mask_a, mask_b,
        why did I raise to exp here again? # TODO: why exp in mask 
        """
    '''
    frames = features.shape[0]
    mask_a = np.zeros(features.shape)
    mask_b = np.zeros(features.shape)
    for i in range(frames):
        mask_a_means = np.exp(a_mean[int(a_ss[i])])
        mask_b_means = np.exp(b_mean[int(b_ss[i])])
        mask_a[i, :] = mask_a_means / ((mask_b_means * alpha) + mask_a_means)
        mask_b[i, :] = mask_b_means / (mask_b_means + (mask_a_means * alpha))
    return mask_a, mask_b

    # @staticmethod TODO: add hard mask
    # def getHardMask(features, model01: inputParams, model02: inputParams):
    #     frames = features.shape[0]
    #     mask01 = np.zeros(features.shape)
    #     mask02 = np.zeros(features.shape)
    #     for i in range(frames):
    #         mask01_means = model01.m[model01.ss[i]]
    #         mask02_means = model02.m[model02.ss[i]]
    #         mask_test = (mask01_means > mask02_means).astype(int)
    #         mask01[i, :] = mask_test
    #         mask02[i, :] = ((mask_test - 1) * -1)
    #     return mask01, mask02