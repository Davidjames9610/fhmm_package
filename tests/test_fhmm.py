from src.fhmm_davidjames9610.fhmm import FHMM
import pytest
import librosa
import unittest
from unittest import mock
import os
from tests.make_data import create_data
import logging
import numpy as np 

# Configure logging
logging.basicConfig(level=logging.INFO)  # Set the logging level as needed

# Function to set up things before other tests


class TestFHMM(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

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
            print('removing file: ' + str(file))
            os.remove(output_dir + file)

    def test_fhmm_init(self):
        logging.info("This is an info message: setUpClass")
        my_fhmm = FHMM(2,2)
        assert my_fhmm.init == True


if __name__ == "__main__":
    unittest.main()


