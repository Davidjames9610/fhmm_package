from src.fhmm_davidjames9610.fhmm import FHMM
import pytest
import logging

def test_fhmm_init():
    print('hi')
    my_fhmm = FHMM(2,2)
    assert my_fhmm.init == True


