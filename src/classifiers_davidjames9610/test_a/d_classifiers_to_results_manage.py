#%%
import sys
# sys.path.append("/Users/david/Documents/code/fhmm/v1")
from importlib import reload
from src.classifiers_davidjames9610.test_a.config import config_location
import importlib
import src.classifiers_davidjames9610.test_a.config as base_config
import pickle
import src.classifiers_davidjames9610.test_a.d_classifiers_to_results as d_classifiers_to_results
import src.misc_davidjames9610.utils as utils
import time
import pickle

#%%

def run_script():
    reload(base_config)
    config = importlib.import_module(base_config.config_location)
    reload(config)
    print(config.basedir)

    reload(d_classifiers_to_results)
    sls = pickle.load(open(config.samples_labels, 'rb'))
    features = utils.folder_pickles_to_dict(config.basedir + '/features', 'mfcc')
    classifiers = utils.folder_pickles_to_dict(config.basedir + '/classifiers', 'GMMHMM')

    plot_cms = True
    save_cms = True

    include_buffer = False
    if include_buffer:
        d_classifiers_to_results.include_buffer_in_features(features, buffer_length=250, buffer_step=125)
        pickle.dump(features, open(config.features, 'wb'))

    start_time = time.time()

    results = {}

    result_types = [
        'classification_annotations',
    ]

    result_type = result_types[0]

    d_classifiers_to_results.get_classification_results(
        features, classifiers, sls, config.basedir, result_type=result_type, plot_cm=plot_cms, save_plots=save_cms,
        new_results=False)

    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.6f} seconds")


#%%
if __name__ == "__main__":
    run_script()
