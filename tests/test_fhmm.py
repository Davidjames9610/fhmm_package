from src.fhmm_davidjames9610.fhmm import FHMM
import pytest
import librosa
import unittest
from unittest import mock
import os
from tests.make_data import create_data
import logging
import numpy as np 
import matplotlib.pyplot as plt

import warnings

# Mute DeprecationWarning

# Configure logging
logging.basicConfig(level=logging.INFO)  # Set the logging level as needed

# Function to set up things before other tests


class TestFHMM(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # remove csv files
        logging.info("This is an info message: setUpClass")

        output_dir = "./tests/test_data/"
        for file in os.listdir(output_dir):
            print('removing file: ' + str(file))
            os.remove(output_dir + file)
    
    def setUp(self) -> None:
        # create csv files
        create_data()
        # load data
        output_dir = "./tests/test_data/"
        csv_files = [file for file in os.listdir(output_dir) if file.endswith('.csv')]
        loaded_arrays = {}
        # Load each CSV file into a NumPy array and store in the dictionary
        for file in csv_files:
            file_path = os.path.join(output_dir, file)
            array_name = file.split('.')[0]  # Use file name without extension as array name
            loaded_arrays[array_name] = np.genfromtxt(file_path, delimiter=',')
        self.features = loaded_arrays

    def tearDown(self):
        # Cleanup tasks to be performed after tests
        print("Running cleanup tasks...")
        # Your cleanup code here
        output_dir = "./tests/test_data/"
        for file in os.listdir(output_dir):
            if file.endswith('.csv'):
                print('removing file: ' + str(file))
                os.remove(output_dir + file)

    def test_fhmm_init(self):
        my_fhmm = FHMM(2,2)
        assert my_fhmm.init == True

    def test_fhmm_fit_run(self):
        # does fit even run ? 
        my_fhmm = FHMM(2,2)
        my_fhmm.fit(self.features['whale_features'], self.features['robin_features'])
        assert my_fhmm.hmm_combined is not None

    def test_fhmm_decode(self):
        # does decode work ?
        my_fhmm = FHMM(2,2)
        my_fhmm.fit(self.features['whale_features'], self.features['robin_features'])
        log_prob, [ss_a, ss_b] = my_fhmm.decode(self.features['combined_features'])
        assert log_prob is not None
        assert len(ss_a) == len(ss_b)

    def test_fhmm_state_sequence_should_be_close_to_individual_state_sequences(self):
        # if we decode the individual HMMs on the features they should be close to the
        # states that the FHMM finds 
        my_fhmm = FHMM(5,5)
        my_fhmm.fit(self.features['whale_features'], self.features['robin_features'])
        log_prob, [ss_a, ss_b] = my_fhmm.decode(self.features['combined_features'])
        # individual on individual
        _, ss_a_ind = my_fhmm.hmm_a.decode(self.features['whale_features'])
        _, ss_b_ind = my_fhmm.hmm_b.decode(self.features['robin_features'])
        # individual on mixed
        _, ss_a_mixed = my_fhmm.hmm_a.decode(self.features['combined_features'])
        _, ss_b_mixed = my_fhmm.hmm_b.decode(self.features['combined_features'])

        acc_a = np.sum(ss_a == ss_a_ind) / len(ss_a)
        acc_b = np.sum(ss_b == ss_b_ind) / len(ss_b)

        plt.plot(ss_a_mixed, '--', c='grey', label='hmm')
        plt.plot(ss_a, '-', c='lightcoral', label='fhmm')
        plt.plot(ss_a_ind, '-', c='black', label='true')

        plt.legend(loc='upper right', facecolor='white',framealpha=1)
        plt.savefig('./tests/output/whale_states.png')
        plt.close()

        plt.plot(ss_b_mixed, '--', c='grey', label='hmm')
        plt.plot(ss_b, '-', c='lightcoral', label='fhmm')
        plt.plot(ss_b_ind, '-', c='black', label='true')

        plt.legend(loc='upper right', facecolor='white',framealpha=1)
        plt.savefig('./tests/output/robin_states.png')
        plt.close()


if __name__ == "__main__":

    unittest.main()


