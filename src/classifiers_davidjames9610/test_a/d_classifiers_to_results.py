import os
import time
from importlib import reload

from hmmlearn.base import BaseHMM
import numpy as np
from scipy.stats import stats
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import src.misc_davidjames9610.decode_combine as dc
import src.misc_davidjames9610.utils as utils


def get_classification_results(features, classifiers, sls, basedir, result_type, plot_cm=True, save_plots=True,
                               quick=False, save_results=True, new_results=False):

    results = {}  # one for each process method

    output_dir = basedir + '/plots/' + result_type + '/'
    utils.create_directory_if_not_exists(output_dir, clean_dir=False)

    print('type:', result_type)

    for classifier_key in classifiers:

        classifier = classifiers[classifier_key]

        all_features_for_classifier = classifier['features']
        classifier_type = classifier['type']
        results[classifier_type] = {}
        print('For classifier:', classifier_type)

        feature_keys_saved = []
        if not new_results:
            is_dir = os.path.isdir(basedir + '/results/' + result_type + '/' + classifier_type + '/')
            if is_dir:
                for file_name in os.listdir(basedir + '/results/' + result_type + '/' + classifier_type + '/'):
                    clean_file_name = file_name.replace('.pickle', '')
                    feature_keys_saved.append(clean_file_name)
                print('already saved:', feature_keys_saved)

        for feature_key in all_features_for_classifier:

            if feature_key not in feature_keys_saved:

                print('  Testing for:', feature_key)

                cv_index = 0  # todo think about updating this cv index for testing

                curr_features = features[feature_key]['val_features'][cv_index]
                curr_label = features[feature_key]['val_label'][cv_index]
                curr_classifiers = classifier['trained_classifiers'][feature_key]
                test_labels = []
                start_time_inner = time.time()

                performance_metrics = None

                if result_type == 'classification_annotations':

                    for feat in curr_features:
                        speakers_scores = []
                        for speaker in sls['labels_set']:
                            speaker_hmm: BaseHMM = curr_classifiers[speaker]
                            speakers_scores.append(speaker_hmm.score(feat))
                        arg_max_speaker = np.argmax(speakers_scores)
                        test_labels.append(arg_max_speaker)

                    performance_metrics = utils.get_performance_metrics(np.array(test_labels), curr_label,
                                                                        labels=list(sls['num_to_label'].keys()))

                elif result_type == 'classification_buffer':

                    # curr_features = features[feature_key]['val_features'][cv_index]
                    # curr_labels = features[feature_key]['val_label'][cv_index]
                    # curr_classifiers = classifier['trained_classifiers'][feature_key]
                    buffer_features = features[feature_key]['buffer_features']
                    # buffer_labels = features[feature_key]['buffer_labels']
                    buffer_labels_mode = features[feature_key]['buffer_labels_mode']

                    test_labels = []

                    for feat in buffer_features:
                        speakers_scores = []
                        for speaker in sls['labels_set']:
                            speaker_hmm: BaseHMM = curr_classifiers[speaker]
                            speakers_scores.append(speaker_hmm.score(feat))
                        arg_max_speaker = np.argmax(speakers_scores)
                        test_labels.append(arg_max_speaker)

                    performance_metrics = utils.get_performance_metrics(buffer_labels_mode, np.array(test_labels),
                                                                        list(sls['num_to_label'].keys()))

                elif result_type == 'classification_annotations_valg':

                    long_labels = [curr_label[i] * np.ones(len(curr_features[i])) for i in range(len(curr_label))]
                    labels_true = np.concatenate(long_labels)

                    if 'fhmm' in classifier_type:
                        classifiers_to_combine = [curr_classifiers[speaker_key].hmm_combined for speaker_key in
                                                  curr_classifiers]
                    else:
                        classifiers_to_combine = [curr_classifiers[speaker_key] for speaker_key in
                                                  curr_classifiers]

                    combined_model = None
                    if 'fhmm' in classifier_type or 'GaussianHMM' in classifier_type:
                        combined_model = dc.DecodeCombineGaussian(classifiers_to_combine)
                    elif 'GMMHMM' in classifier_type:
                        combined_model = dc.DecodeCombineGMMHMM(classifiers_to_combine)

                    _, labels_predicted, val_log_prob = combined_model.decode_hmmlearn(np.concatenate(curr_features))

                    # CM
                    performance_metrics = utils.get_performance_metrics(labels_true, labels_predicted,
                                                                        list(sls['num_to_label'].keys()))

                elif result_type == 'classification_buffer_valg':

                    buffer_features = features[feature_key]['buffer_features']  # [:20]
                    buffer_labels = features[feature_key]['buffer_labels']  # [:20]
                    buffer_labels_mode = features[feature_key]['buffer_labels_mode']  # [:20]

                    # long_labels = [curr_labels[i] * np.ones(len(curr_features[i])) for i in range(len(curr_labels))]
                    # labels_true = np.concatenate(long_labels)

                    if 'fhmm' in classifier_type:
                        classifiers_to_combine = [curr_classifiers[speaker_key].hmm_combined for speaker_key in
                                                  curr_classifiers]
                    else:
                        classifiers_to_combine = [curr_classifiers[speaker_key] for speaker_key in
                                                  curr_classifiers]

                    combined_model = None
                    if 'fhmm' in classifier_type or 'GaussianHMM' in classifier_type:
                        combined_model = dc.DecodeCombineGaussian(classifiers_to_combine)
                    elif 'GMMHMM' in classifier_type:
                        combined_model = dc.DecodeCombineGMMHMM(classifiers_to_combine)

                    labels_predicted = []
                    labels_predicted_mode = []
                    for feat in buffer_features:
                        _, feat_labels_predicted, val_log_prob = combined_model.decode_hmmlearn(feat)
                        labels_predicted.append(feat_labels_predicted)
                        labels_predicted_mode.append(stats.mode(feat_labels_predicted, keepdims=True).mode[0])

                    # long input # todo review this
                    # buffer_labels_concat = np.concatenate(buffer_labels)
                    # labels_predicted_concat = np.concatenate(labels_predicted)
                    # min_length = np.min([len(buffer_labels_concat), len(labels_predicted_concat)])

                    # mode input
                    min_length = np.min([len(labels_predicted_mode), len(buffer_labels_mode)])

                    performance_metrics = utils.get_performance_metrics(
                        labels_predicted_mode[:min_length],
                        buffer_labels_mode[:min_length],
                        labels=list(sls['num_to_label'].keys()))

                disp = ConfusionMatrixDisplay(confusion_matrix=performance_metrics['cm'],
                                              display_labels=list(sls['num_to_label'].keys()))
                disp.plot(cmap=plt.cm.Blues, values_format='.2f')
                plt.title(classifier_type + '_' + feature_key)
                if save_plots:
                    plt.savefig(output_dir + classifier_type + '_' + feature_key + '.png')
                if plot_cm:
                    plt.show()
                plt.close()

                end_time_inner = time.time()
                performance_metrics['time'] = end_time_inner - start_time_inner
                results[classifier_type][feature_key] = performance_metrics

                if quick:
                    break
        # save here
        utils.dict_to_folder_pickles(basedir + '/results/' + result_type + '/' + classifier_type,
                                     results[classifier_type])
        if quick:
            break

    return results

def get_classification_buff_results(features, classifiers, sls, basedir,
                                    plot_cm=True, save_plots=True,
                                    classification_type='normal'):
    results = {}  # one for each process method

    output_dir = basedir + '/results/classification/buffer/'
    utils.create_directory_if_not_exists(output_dir)

    # CLASSIFICATION
    for classifier in classifiers:

        all_features_for_classifier = classifier['features']
        classifier_type = classifier['type']
        results[classifier_type] = {}
        print('For classifier:', classifier_type)

        for feature_key in all_features_for_classifier:
            print('  Testing for:', feature_key)

            start_time_inner = time.time()

            cv_index = 0  # todo think about updating this cv index for testing

            curr_features = features[feature_key]['val_features'][cv_index]
            curr_labels = features[feature_key]['val_label'][cv_index]
            curr_classifiers = classifier['trained_classifiers'][feature_key]
            buffer_features = features[feature_key]['buffer_features']
            buffer_labels = features[feature_key]['buffer_labels']
            buffer_labels_mode = features[feature_key]['buffer_labels_mode']

            test_labels = []

            for feat in buffer_features:
                speakers_scores = []
                for speaker in sls['labels_set']:
                    speaker_hmm: BaseHMM = curr_classifiers[speaker]
                    speakers_scores.append(speaker_hmm.score(feat))
                arg_max_speaker = np.argmax(speakers_scores)
                test_labels.append(arg_max_speaker)

            performance_metrics = utils.get_performance_metrics(buffer_labels_mode, np.array(test_labels),
                                                                list(sls['num_to_label'].keys()))

            disp = ConfusionMatrixDisplay(confusion_matrix=performance_metrics['cm'],
                                          display_labels=list(sls['num_to_label'].keys()))
            disp.plot(cmap=plt.cm.Blues, values_format='.2f')
            plt.title(classifier_type + '_' + feature_key)
            if save_plots:
                plt.savefig(output_dir + classifier_type + '_' + feature_key + '.png')
            if plot_cm:
                plt.show()
            plt.close()

            end_time_inner = time.time()
            performance_metrics['time'] = end_time_inner - start_time_inner
            results[classifier_type][feature_key] = performance_metrics
            if quick: break
        if quick: break
    return results


# similar to above, except run V-alg over concat input sequence
def get_classification_valg_results(features, classifiers, sls, basedir, plot_cm=True, save_plots=True):
    results = {}  # one for each process method

    count = 0

    output_dir = basedir + '/results/classification-valg/normal/'
    utils.create_directory_if_not_exists(output_dir)

    # CLASSIFICATION
    for classifier in classifiers:

        start_time = time.time()

        all_features_for_classifier = classifier['features']
        classifier_type = classifier['type']

        if 'fhmm' in classifier_type or 'GaussianHMM' in classifier_type:  # update this

            results[classifier_type] = {}
            start_time_inner = time.time()

            print('For classifier:', classifier_type)

            # combined hmm approach, still Viterbi, this assumes I have longish real input data

            # decode_combine_results = d_classifiers_to_results.decode_combine_results()
            # todo include annotation for spectrogram

            for feature_key in all_features_for_classifier:

                print('testing for feature type: ', feature_key)
                cv_index = 0
                curr_features = features[feature_key]['val_features'][cv_index]
                curr_labels = features[feature_key]['val_label'][cv_index]
                curr_classifiers = classifier['trained_classifiers'][feature_key]

                long_labels = [curr_labels[i] * np.ones(len(curr_features[i])) for i in range(len(curr_labels))]
                labels_true = np.concatenate(long_labels)

                if 'fhmm' in classifier_type:
                    classifiers_to_combine = [curr_classifiers[speaker_key].hmm_combined for speaker_key in
                                              curr_classifiers]
                else:
                    classifiers_to_combine = [curr_classifiers[speaker_key] for speaker_key in
                                              curr_classifiers]

                combined_model = dc.DecodeCombineGaussian(classifiers_to_combine)

                _, labels_predicted, val_log_prob = combined_model.decode_hmmlearn(np.concatenate(curr_features))

                # CM
                performance_metrics = utils.get_performance_metrics(labels_true, labels_predicted,
                                                                    list(sls['num_to_label'].keys()))

                disp = ConfusionMatrixDisplay(confusion_matrix=performance_metrics['cm'],
                                              display_labels=list(sls['num_to_label'].keys()))
                disp.plot(cmap=plt.cm.Blues, values_format='.2f')
                plt.title(classifier_type + '_' + feature_key)
                if save_plots:
                    plt.savefig(output_dir + classifier_type + '_' + feature_key + '.png')
                if plot_cm:
                    plt.show()
                plt.close()

                end_time_inner = time.time()
                performance_metrics['time'] = end_time_inner - start_time_inner
                results[classifier_type][feature_key] = performance_metrics

                if quick: break
        if quick: break

    return results


def get_classification_valg_buffer_results(features, classifiers, sls, basedir, plot_cm=True, save_plots=True):
    results = {}  # one for each process method

    count = 0

    output_dir = basedir + '/results/classification-valg/buffer/'
    utils.create_directory_if_not_exists(output_dir)

    # CLASSIFICATION
    for classifier in classifiers:

        start_time = time.time()

        all_features_for_classifier = classifier['features']
        classifier_type = classifier['type']

        if 'fhmm' in classifier_type or 'GaussianHMM' in classifier_type:

            results[classifier_type] = {}

            print('For classifier:', classifier_type)

            # combined hmm approach, still Viterbi, this assumes I have longish real input data

            # decode_combine_results = d_classifiers_to_results.decode_combine_results()
            # todo include annotation for spectrogram

            for feature_key in all_features_for_classifier:

                print('testing for feature type: ', feature_key)
                cv_index = 0
                curr_features = features[feature_key]['val_features'][cv_index]
                curr_labels = features[feature_key]['val_label'][cv_index]
                curr_classifiers = classifier['trained_classifiers'][feature_key]

                buffer_features = features[feature_key]['buffer_features']  # [:20]
                buffer_labels = features[feature_key]['buffer_labels']  # [:20]
                buffer_labels_mode = features[feature_key]['buffer_labels_mode']  # [:20]

                # long_labels = [curr_labels[i] * np.ones(len(curr_features[i])) for i in range(len(curr_labels))]
                # labels_true = np.concatenate(long_labels)

                if 'fhmm' in classifier_type:
                    classifiers_to_combine = [curr_classifiers[speaker_key].hmm_combined for speaker_key in
                                              curr_classifiers]
                else:
                    classifiers_to_combine = [curr_classifiers[speaker_key] for speaker_key in
                                              curr_classifiers]

                combined_model = dc.DecodeCombineGaussian(classifiers_to_combine)

                labels_predicted = []
                labels_predicted_mode = []
                for feat in buffer_features:
                    _, feat_labels_predicted, val_log_prob = combined_model.decode_hmmlearn(feat)
                    labels_predicted.append(feat_labels_predicted)
                    labels_predicted_mode.append(stats.mode(feat_labels_predicted, keepdims=True).mode[0])

                # long input
                buffer_labels_concat = np.concatenate(buffer_labels)
                labels_predicted_concat = np.concatenate(labels_predicted)
                min_length = np.min([len(buffer_labels_concat), len(labels_predicted_concat)])

                # mode input
                min_length = np.min([len(labels_predicted_mode), len(buffer_labels_mode)])

                performance_metrics = utils.get_performance_metrics(
                    labels_predicted_mode[:min_length],
                    buffer_labels_mode[:min_length],
                    labels=list(sls['num_to_label'].keys()))

                disp = ConfusionMatrixDisplay(confusion_matrix=performance_metrics['cm'],
                                              display_labels=list(sls['num_to_label'].keys()))
                disp.plot(cmap=plt.cm.Blues, values_format='.2f')
                plt.title(classifier_type + '_' + feature_key)
                if save_plots:
                    plt.savefig(output_dir + classifier_type + '_' + feature_key + '.png')
                if plot_cm:
                    plt.show()
                plt.close()

                end_time_inner = time.time()
                performance_metrics['time'] = end_time_inner - start_time_inner
                results[classifier_type][feature_key] = performance_metrics
                if quick: break
        if quick: break

    return results


# updates feature dictionary to include buffered features and labels
def include_buffer_in_features(features, buffer_length=500, buffer_step=250):
    for feature_key in features:
        print('  buffering for:', feature_key)

        cv_index = 0

        curr_features = features[feature_key]['val_features'][cv_index]
        curr_labels = features[feature_key]['val_label'][cv_index]

        long_labels = [curr_labels[i] * np.ones(len(curr_features[i])) for i in range(len(curr_labels))]
        labels_true = np.concatenate(long_labels)

        concat_features = np.concatenate(curr_features)

        _, buffer_features = utils.buffer_features(concat_features, buffer_length, buffer_step)
        buffer_labels = utils.buffer(labels_true, buffer_length, buffer_step, opt='nodelay')
        buffer_labels_mode = [stats.mode(label_win, keepdims=True).mode[0] for label_win in buffer_labels]

        features[feature_key]['buffer_features'] = buffer_features
        features[feature_key]['buffer_labels'] = buffer_labels
        features[feature_key]['buffer_labels_mode'] = buffer_labels_mode
        # break
