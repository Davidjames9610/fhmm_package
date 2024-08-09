from hmmlearn.base import BaseHMM
from hmmlearn.hmm import GaussianHMM

import src.misc_davidjames9610.decode_combine as dc
import src.fhmm_davidjames9610.fhmm as fhmm

import matplotlib.pyplot as plt

import numpy as np

# insert for adaptive noise handling
# update noise hmm to bayesian / HDP ?

class NoiseAdaptiveHMM:
    def __init__(self,
                 base_classifiers: [BaseHMM],
                 noise_features,
                 label_set,
                 noise_hmm_components=3):

        self.base_classifiers = base_classifiers
        self.current_noise_features = noise_features
        self.label_set = label_set

        self.noise_hmm: BaseHMM | None = None
        self.dc_model: dc.DecodeCombineGaussian | None = None
        self.noise_hmm_components = noise_hmm_components
        self.update_model(noise_features, is_update=False)

    def update_model(self, noise_features, is_update=True):

        if is_update:
            print('model updating to changing noise conditions')
        else:
            print('init noise adaptive hmm')

        self.current_noise_features = noise_features

        fhmm_classifiers = {}
        for label in self.label_set:
            curr_classifier = self.base_classifiers[label]
            fhmm_classifier = fhmm.FHMM(curr_classifier.n_components, n_components_b=self.noise_hmm_components)
            fhmm_classifier.fit_given_signal_hmm(curr_classifier, noise_features)
            fhmm_classifiers[label] = fhmm_classifier

        classifiers_to_combine = [fhmm_classifiers[speaker_key].hmm_combined for speaker_key in
                                  fhmm_classifiers]

        noise_hmm = GaussianHMM(
            n_components=self.noise_hmm_components,
            covariance_type='diag')

        noise_hmm.fit(noise_features)

        self.noise_hmm = noise_hmm

        classifiers_to_combine.append(noise_hmm)

        combined_model = dc.DecodeCombineGaussian(classifiers_to_combine)

        self.dc_model = combined_model


def sliding_windows(data, window_size, step_size, na_hmm: NoiseAdaptiveHMM, mean_log_prob, threshold=1.2, re_train_buffer=10):
    output = {
        'noise_data': []
    }
    windows = []
    log_probs = []
    std_probs = []
    states_decoded = np.zeros(len(data))
    window_indices = []

    train_counter = 0

    for i in range(0, len(data) - window_size + 1, step_size):
        window = data[i:i+window_size, :]

        # do stuff here
        _, test_pred, log_prob = na_hmm.dc_model.decode_hmmlearn(window)
        log_probs.append(log_prob)
        states_decoded[i:i+window_size] = test_pred
        windows.append(window)
        window_indices.append(i)
        std_probs.append(np.std(log_probs[-3:]))

        # avoid re-training a lot
        if train_counter > 0:
            train_counter -= 1

        # if likelihood drops bellow threshold then re-train noise hmm and update other hmms
        # maybe don't compare to complete std mean here ?
        if np.mean(log_probs[-10:]) < mean_log_prob * threshold:
            if(std_probs[-1] < np.mean(std_probs)) and train_counter == 0:
                # use last 3 windows to train noise hmm
                # data[i:i + (step_size * 3)]
                start_index = window_indices[-3]
                noise_data = data[start_index:start_index + (step_size * 10)]
                na_hmm.update_model(noise_data)

                output['noise_data'].append(noise_data)

                # start buffer to avoid a lot of re-training
                train_counter = re_train_buffer

    output['windows'] = np.array(windows)
    output['prob'] = log_probs
    output['states'] = states_decoded
    return output

def smooth_labels(labels, window_size=50, step_size=10, diff_size=20):

    smoothy_labels = labels.copy()
    arg_max = []
    arg_max_index = []

    for start in range(0, len(labels) - window_size + 1, step_size):
        end = start + window_size
        window = labels[start:end]

        unique_elements, counts = np.unique(window, return_counts=True)
        max_count_index = np.argmax(counts)
        dominant_label = unique_elements[max_count_index]
        arg_max.append(dominant_label)
        arg_max_index.append(start)

    changes = []
    changes_index = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            changes.append(labels[i])
            changes_index.append(i)

    fwd = changes[2]
    curr = changes[1]
    prev = changes[0]
    for i in range(2, len(changes)):
        fwd = changes[i]
        curr = changes[i - 1]
        prev = changes[i - 2]
        index_curr = changes_index[i - 1]
        index_fwd = changes_index[i]
        index_prev = changes_index[i - 2]
        diff = index_fwd - index_curr
        # two cases, if quick switch back to original state then just set state to backwards
        if prev != curr:
            if diff < diff_size and prev == fwd:
                smoothy_labels[index_curr: index_fwd] = np.ones(diff) * prev
                changes[i - 1] = prev
            elif diff < diff_size and prev != fwd and curr != fwd:
                arg_max_index_cur = np.argmin(np.abs(np.array(arg_max_index) - index_curr))
                arg_max_index_fwd = np.argmin(np.abs(np.array(arg_max_index) - index_fwd))
                if (arg_max[arg_max_index_cur] == arg_max[arg_max_index_fwd]):
                    smoothy_labels[index_curr: index_fwd] = np.ones(diff) * fwd
                    changes[i - 1] = fwd
    return smoothy_labels

def find_label_changes(labels):
    changes = []
    changes.append((0, labels[0]))
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            changes.append((i, labels[i]))
    return changes

def plot_spectrogram(features, true_labels, pred_labels, label_type, label_abr):

    times = np.arange(features.shape[0])
    frequencies = np.arange(features.shape[1])
    freq_max = features.shape[1]
    true_changes = find_label_changes(true_labels)

    pred_changes = find_label_changes(pred_labels)

    for index, label in true_changes:
        t = plt.text(times[index], frequencies[-7], label_abr[label], color='black', fontsize=10, verticalalignment='bottom')
        t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='red', linewidth=0))
        plt.vlines(times[index], ymin=freq_max / 2, ymax=freq_max, color='black', linestyles='dashed', linewidth=1)

    for index, label in pred_changes:
        if(index + 8 < len(times)):
            t = plt.text(times[index + 8], frequencies[3], label_abr[label], color='red', fontsize=10, verticalalignment='bottom')
        else:
            t = plt.text(times[index], frequencies[3], label_abr[label], color='red', fontsize=10,
                         verticalalignment='bottom')
        t.set_bbox(dict(facecolor='white', alpha=0.8, edgecolor='red', linewidth=0))
        plt.vlines(times[index], ymin=0, ymax=freq_max / 2, color='red', linestyles='solid', linewidth=2)

    plt.pcolormesh(features.T, cmap='viridis')
    plt.ylabel('F bin')
    plt.xlabel('T bin')
    legend_labels = [label_abr[label] + ' - ' + label_type[label].title() for label in label_type]
    # plt.legend(legend_labels, loc='upper right', bbox_to_anchor=(1, 1), facecolor='white', framealpha=1, handlelength=0)
    # plt.colorbar(label='Intensity [dB]')
    # plt.title('Annotated Spectrogram')
    plt.show()
