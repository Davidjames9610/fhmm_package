from src.fhmm_davidjames9610.fhmm import FHMM
import pytest
import librosa
import unittest
from unittest import mock
import os
from tests.make_data import get_feature
from tests import test_fhmm_utils
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
        # create_data()
        # # load data 
        # output_dir = "./tests/test_data/"
        # csv_files = [file for file in os.listdir(output_dir) if file.endswith('.csv')]
        # loaded_arrays = {}
        # # Load each CSV file into a NumPy array and store in the dictionary
        # for file in csv_files:
        #     file_path = os.path.join(output_dir, file)
        #     array_name = file.split('.')[0]  # Use file name without extension as array name
        #     loaded_arrays[array_name] = np.genfromtxt(file_path, delimiter=',')
        # self.features = loaded_arrays

        file_a = librosa.example('humpback')
        y_a, sr = librosa.load(file_a)
        y_a = librosa.util.normalize(y_a)
        self.sr = sr

        file_b = librosa.example('robin')
        y_b, sr = librosa.load(file_b)
        y_b = librosa.util.normalize(y_b)

        # split whale to same length as robin 
        oft = 80000
        y_a = y_a[oft:len(y_b) + oft]
        self.whale_audio = y_a
        self.robin_audio = y_b

        # combine sounds in time domain
        self.whale_robin_audio = self.whale_audio + self.robin_audio

        nfft = 128
        self.whale_features = get_feature(self.whale_audio, nfft=nfft)
        self.robin_features = get_feature(self.robin_audio, nfft=nfft)
        self.whale_robin_features = get_feature(self.whale_robin_audio, nfft=nfft)

        # create noise for combining
        snr_1 = test_fhmm_utils.get_noise_avg_watts(self.whale_audio, 15)
        snr_2 = test_fhmm_utils.get_noise_avg_watts(self.whale_audio, 3)
        snr_3 = test_fhmm_utils.get_noise_avg_watts(self.whale_audio, 7)

        gauss_noise, ss = test_fhmm_utils.generate_gaussian_noise(len(self.whale_audio),snr_1 , snr_2, snr_3)




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
        my_fhmm = FHMM(2,2)
        my_fhmm.fit(self.features['whale_features'], self.features['robin_features'])
        assert my_fhmm.hmm_combined is not None

    def test_fhmm_decode(self):
        my_fhmm = FHMM(2,2)
        my_fhmm.fit(self.features['whale_features'], self.features['robin_features'])
        log_prob, [ss_a, ss_b] = my_fhmm.decode(self.features['combined_features'])
        assert log_prob is not None
        assert len(ss_a) == len(ss_b)

    def test_fhmm_state_sequence_should_be_close_to_individual_state_sequences(self):
        # given a true state sequence of a trained hmm on the features it was trained on,
        # we should be able to recover something close to this sequence using the fhmm 
        # on mixed features 
        my_fhmm = FHMM(3,3)
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

        assert acc_a > 0.90
        assert acc_b > 0.90

    def test_fhmm_state_sequence_should_be_close_to_individual_state_sequences_noise(self):
        # this is similar test to above except noise is used instead of a different sound


        test_fhmm_utils.generate_gaussian_noise()


if __name__ == "__main__":

    unittest.main()


