import math

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


def test_log_binomial_zero_n_zero_k_is_zero():
    assert math_utils.log_binomial(0, 0, 0.5) == 0.0


def test_log_binomial_known_value():
    # log(C(4,2) * 0.5^4) = log(6/16)
    assert math.isclose(math_utils.log_binomial(4, 2, 0.5), math.log(6 / 16), rel_tol=1e-12)


def test_log_binomial_clamps_extreme_p():
    # p=0/1 would be log(0) without clamping; result must be finite.
    assert np.isfinite(math_utils.log_binomial(10, 0, 0.0))
    assert np.isfinite(math_utils.log_binomial(10, 10, 1.0))


def test_log_binomial_invalid_k_returns_neg_inf():
    assert math_utils.log_binomial(5, -1, 0.5) == -math.inf
    assert math_utils.log_binomial(5, 6, 0.5) == -math.inf
    assert math_utils.log_binomial(0, 1, 0.5) == -math.inf
