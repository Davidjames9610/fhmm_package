import os, shutil
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
def create_directory_if_not_exists(directory_path, clean_dir=True):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists, removing old files: ", clean_dir)
        if clean_dir:
            for file_name in os.listdir(directory_path):
                file_path = os.path.join(directory_path, file_name)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print('Failed to delete %s. Reason: %s' % (file_path, e))

def folder_pickles_to_dict(complete_dir, file_part_to_include=None, list_to_include=None):
    some_dict = {}
    for file_name in os.listdir(complete_dir):
        clean_file_name = file_name.replace('.pickle', '')
        if file_part_to_include is not None:
            if file_name.__contains__(file_part_to_include):
                print('loading', file_name)
                file_path = os.path.join(complete_dir, file_name)
                some_dict[clean_file_name] = (pickle.load(open(file_path, 'rb')))
        elif list_to_include is not None:
            if file_name.replace('.pickle', '') in list_to_include:
                print('loading', file_name)
                file_path = os.path.join(complete_dir, file_name)
                some_dict[clean_file_name] = (pickle.load(open(file_path, 'rb')))
        else:
            print('loading', file_name)
            file_path = os.path.join(complete_dir, file_name)
            some_dict[clean_file_name] = (pickle.load(open(file_path, 'rb')))

    sorted_dict = {k: some_dict[k] for k in sorted(some_dict)}
    return sorted_dict

def dict_to_folder_pickles(folder_name, some_dict):
    create_directory_if_not_exists(folder_name, clean_dir=False)
    for key in some_dict:
        print('saving / updating ', key)
        pickle.dump(some_dict[key], open(folder_name + '/' + key + '.pickle', 'wb'))

def buffer(x, n, p=0, opt=None):
    """
    Mimic MATLAB routine to generate buffer array

    MATLAB docs here: https://se.mathworks.com/help/signal/ref/buffer.html

    Parameters
    ----------
    x: ndarray
        Signal array
    n: int
        Number of data segments
    p: int
        Number of values to overlap
    opt: str
        Initial condition options. default sets the first `p` values to zero,
        while 'nodelay' begins filling the buffer immediately.

    Returns
    -------
    result : (n,n) ndarray
        Buffer array created from x
    """
    import numpy as np

    if opt not in [None, 'nodelay']:
        raise ValueError('{} not implemented'.format(opt))

    i = 0
    first_iter = True
    while i < len(x):
        if first_iter:
            if opt == 'nodelay':
                # No zeros at array start
                result = x[:n]
                i = n
            else:
                # Start with `p` zeros
                result = np.hstack([np.zeros(p), x[:n - p]])
                i = n - p
            # Make 2D array and pivot
            result = np.expand_dims(result, axis=0).T
            first_iter = False
            continue

        # Create next column, add `p` results from last col if given
        col = x[i:i + (n - p)]
        if p != 0:
            col = np.hstack([result[:, -1][-p:], col])
        i += n - p

        # Append zeros if last row and not length `n`
        if len(col) < n:
            col = np.hstack([col, np.zeros(n - len(col))])

        # Combine result with next row
        result = np.hstack([result, np.expand_dims(col, axis=0).T])

    return result.T
def buffer_features(x, n, p=0, opt='nodelay'):
    """
    Mimic MATLAB routine to generate buffer array

    MATLAB docs here: https://se.mathworks.com/help/signal/ref/buffer.html

    Parameters
    ----------
    x: ndarray
        Signal array
    n: int
        Number of data segments
    p: int
        Number of values to overlap
    opt: str
        Initial condition options. default sets the first `p` values to zero,
        while 'nodelay' begins filling the buffer immediately.

    Returns
    -------
    result : (n,n) ndarray
        Buffer array created from x
    """
    import numpy as np

    if opt not in [None, 'nodelay']:
        raise ValueError('{} not implemented'.format(opt))

    n_features = x.shape[1]
    i = 0
    first_iter = True
    return_arr = []
    while i < len(x):
        if first_iter:
            if opt == 'nodelay':
                # No zeros at array start
                result = x[:n, :]
                i = n
            else:
                # Start with `p` zeros
                result = np.vstack([np.zeros((p, n_features)), x[:n - p, :]])
                i = n - p
            # Make 2D array and pivot
            result = np.expand_dims(result.T, axis=0).T
            first_iter = False
            continue

        # Create next column, add `p` results from last col if given
        col = x[i:i + (n - p), :]
        if p != 0:
            col = np.vstack([result[:, :, -1][-p:, :], col])
            return_arr.append(col)
        i += n - p

        # Append zeros if last row and not length `n`
        if len(col) < n:
            col = np.vstack([col, np.zeros((n - len(col), n_features))])
            return_arr.append(col)

        # Combine result with next row
        result = np.concatenate((result, np.expand_dims(col.T, axis=0).T), axis=2)
    return result.T, return_arr

def get_performance_metrics(y_actual, y_hat, labels):

    cm = confusion_matrix(y_hat, y_actual, labels=labels)

    # Number of classes
    num_classes = cm.shape[0]

    # Initialize arrays to hold TP, TN, FP, FN for each class
    TP_classes = np.zeros(num_classes)
    TN_classes = np.zeros(num_classes)
    FP_classes = np.zeros(num_classes)
    FN_classes = np.zeros(num_classes)

    # Calculate TP, TN, FP, FN for each class
    for i in range(num_classes):
        TP_classes[i] = cm[i, i]
        FP_classes[i] = cm[:, i].sum() - cm[i, i]
        FN_classes[i] = cm[i, :].sum() - cm[i, i]
        TN_classes[i] = cm.sum() - (TP_classes[i] + FP_classes[i] + FN_classes[i])

    TP = np.sum([TP_classes[i] for i in range(num_classes)])
    TN = np.sum([TN_classes[i] for i in range(num_classes)])
    FP = np.sum([FP_classes[i] for i in range(num_classes)])
    FN = np.sum([FN_classes[i] for i in range(num_classes)])

    # Sensitivity, hit rate, recall, or true positive rate
    try:
        TPR = TP/(TP+FN)
    except ZeroDivisionError:
        TPR = None

    # Specificity or true negative rate
    try:
        TNR = TN / (TN + FP)
    except ZeroDivisionError:
        TNR = None

    # Precision or positive predictive value
    try:
        PPV = TP / (TP + FP)
    except ZeroDivisionError:
        PPV = None

    # Negative predictive value
    try:
        NPV = TN / (TN + FN)
    except ZeroDivisionError:
        NPV = None

    # Fall out or false positive rate
    try:
        FPR = FP / (FP + TN)
    except ZeroDivisionError:
        FPR = None

    # False negative rate
    try:
        FNR = FN / (TP + FN)
    except ZeroDivisionError:
        FNR = None

    # False discovery rate
    try:
        FDR = FP / (TP + FP)
    except ZeroDivisionError:
        FDR = None

    # Overall accuracy
    try:
        ACC = (TP + TN) / (TP + FP + FN + TN)
    except ZeroDivisionError:
        ACC = None

    return_metrics = {
        'ACC': ACC,
        'PPV': PPV,
        'TPR': TPR,
        'TNR': TNR,
        'FPR': FPR,
        'FNR': FNR,
        'NPV': NPV,
        'FDR': FDR
    }

    return {
        'stats': return_metrics,
        'cm': confusion_matrix(y_hat, y_actual, labels=labels, normalize='true')}

def get_performance_metrics_from_confusion_matrix(cm):
    # Extract TP, FP, FN, and TN from the confusion matrix
    TN, FP, FN, TP = cm.ravel()

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)

    return_metrics = {
        'ACC': ACC,
        'PPV': PPV,
        'TPR': TPR,
        'TNR': TNR,
        'FPR': FPR,
        'FNR': FNR,
        'FDR': FDR,
        'NPV': NPV
    }

    return return_metrics


# Example usage:
# cm = np.array([[50, 10], [5, 100]])
# metrics = get_performance_metrics_from_confusion_matrix(cm)
# print(metrics)


