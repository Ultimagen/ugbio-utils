import numpy as np
from ugbio_core import math_utils


def test_phred():
    assert np.all(math_utils.phred((0.1, 0.01, 0.001)) == np.array([10.0, 20.0, 30.0]))


def test_phred_str():
    assert math_utils.phred_str([0.1, 0.01, 0.001]) == "+5?"


def test_unphred():
    assert np.allclose(math_utils.unphred((10, 20, 30)), np.array([0.1, 0.01, 0.001]))


def test_unphred_str():
    assert np.allclose(math_utils.unphred_str("+5?"), np.array([0.1, 0.01, 0.001]))
