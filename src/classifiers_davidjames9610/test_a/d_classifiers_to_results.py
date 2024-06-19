from hmmlearn.base import BaseHMM
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def get_classification_results(features, classifiers, sls, plot_cm=True):

    results = {}  # one for each process method

    # CLASSIFICATION
    for feature_key in features:

        print('testing for feature type: ', feature_key)
        cv_index = 0    # todo think about updating this cv index for testing

        curr_features = features[feature_key]['val_features'][cv_index]
        curr_label = features[feature_key]['val_label'][cv_index]
        curr_classifiers = classifiers[feature_key]
        test_labels = []

        for feat in curr_features:
            speakers_scores = []
            for speaker in sls['labels_set']:
                speaker_hmm: BaseHMM = curr_classifiers[speaker]
                speakers_scores.append(speaker_hmm.score(feat))
            arg_max_speaker = np.argmax(speakers_scores)
            test_labels.append(arg_max_speaker)

        cm = confusion_matrix(np.array(test_labels), curr_label, labels=list(sls['num_to_label'].keys()),
                              normalize='true')
        if plot_cm:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(sls['num_to_label'].keys()))
            disp.plot(cmap=plt.cm.Blues, values_format='g')
            plt.show()

        results[feature_key] = cm

    return results
