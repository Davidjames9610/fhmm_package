import numpy as np
import bnpy
from hmmlearn.hmm import GaussianHMM
from bnpy import HModel

def get_GroupXData_from_list(features_list):
    features_concat = np.vstack(features_list)
    features_len = [0]
    n_doc = 0
    rolling_total = 0

    for i in range(len(features_list)):
        features_len.append(len(features_list[i]) + rolling_total)
        rolling_total += len(features_list[i])
        n_doc += 1

    features_len = np.array(features_len)

    return bnpy.data.GroupXData(X=features_concat, doc_range=features_len,
                                nDocTotal=n_doc)


bnpy_config = dict(
    goodelbopairs_merge_kwargs=dict(
        m_startLap=10,
        # Set limits to number of merges attempted each lap.
        # This value specifies max number of tries for each cluster
        m_maxNumPairsContainingComp=5,
        # Set "reactivation" limits
        # So that each cluster is eligible again after 10 passes thru dataset
        # Or when it's size changes by 400%
        m_nLapToReactivate=10,
        m_minPercChangeInNumAtomsToReactivate=400 * 0.01,
        # Specify how to rank pairs (determines order in which merges are tried)
        # 'obsmodel_elbo' means rank pairs by improvement to observation model ELBO
        m_pair_ranking_procedure='obsmodel_elbo',
        m_pair_ranking_direction='descending',
    ),
    init_kwargs=dict(
        K=30,
        initname='randexamples',
    ),
    alg_kwargs=dict(
        nLap=30,
        nTask=1, nBatch=1, convergeThr=0.01,
    ),
    hdphmm_kwargs=dict(
        startAlpha=1000.0,  # top-level Dirichlet concentration parameter
        transAlpha=1000,  # trans-level Dirichlet concentration parameter
        hmmKappa=1000,
    ),
    gauss_kwargs=dict(
        sF=1,  # Set prior so E[covariance] = identity
        ECovMat='eye',
    )
)

def get_hmm_learn_from_bnpy(some_model: HModel):
    obs_model = some_model.obsModel
    total_k = obs_model.K
    means = []
    sigmas = []
    for k in range(total_k):
        sigmas.append(np.diag(obs_model.get_covar_mat_for_comp(k)))
        means.append(obs_model.get_mean_for_comp(k))

    means = np.vstack(means)
    sigmas = np.vstack(sigmas)

    A = some_model.allocModel.get_trans_prob_matrix(),
    pi = some_model.allocModel.get_init_prob_vector(),

    # remove states with zero means
    non_zero_indicis = (np.isclose(np.sum(means, axis=1), 0) == False)
    means = means[non_zero_indicis, :]
    sigmas = sigmas[non_zero_indicis, :]
    A = A[0][non_zero_indicis, :]
    A = A[:, non_zero_indicis]
    pi = pi[0][non_zero_indicis]

    # creat hmm
    hmm_bnpy = GaussianHMM(n_components=len(pi), covariance_type='diag', init_params='')
    hmm_bnpy.n_features = means.shape[1]
    hmm_bnpy.transmat_, hmm_bnpy.startprob_, hmm_bnpy.means_ = normalize_matrix(A), normalize_matrix(pi), means
    hmm_bnpy.covars_ = sigmas
    return hmm_bnpy

def normalize_matrix(matrix):
    matrix += 1e-40
    return matrix / np.sum(matrix, axis=(matrix.ndim - 1), keepdims=True)
