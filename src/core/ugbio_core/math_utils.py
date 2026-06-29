from __future__ import annotations

import math

import numpy as np


def safe_divide(numerator: float, denominator: float, return_if_denominator_is_0: int = 0):
    """
    Parameters
    ----------
    numerator : float
        numerator
    denominator : float
        denominator
    return_if_denominator_is_0 : int, optional
        return value whenever denominator is 0 (undivisable)

    Returns
    -------
    if denominator != 0 -> numerator/denominator
    else -> return_if_denominator_is_0
    """
    if denominator == 0:
        return return_if_denominator_is_0

    return numerator / denominator


def phred(p: list[float] | tuple[float] | np.ndarray) -> np.ndarray:
    """
    Transform probablitied to Phred quality scores
    See https://en.wikipedia.org/wiki/Phred_quality_score

    Parameters
    ----------
    p : Union[list[float], tuple[float], np.ndarray]
        List of float probability values

    Returns
    -------
    np.ndarray
        List of float quality values
    """
    q = -10 * np.log10(np.array(p, dtype=float))
    return q


def phred_str(p: list[float] | tuple[float] | np.ndarray) -> str:
    """Convert list of error probabilities to phred-encoded string

    Parameters
    ----------
    p : Union[list[float], tuple[float], np.ndarray]
        List of float probability values

    Returns
    -------
    str
        Basequality string
    """
    q = phred(p)
    return "".join(chr(int(x) + 33) for x in q)


def unphred(q: float | list[int | float] | tuple[int | float] | np.ndarray) -> np.ndarray | float:
    """Transform Phred quality scores to probablities
    See https://en.wikipedia.org/wiki/Phred_quality_score

    Parameters
    ----------
    q : Union[float, list, tuple, np.ndarray]
        List of integer or float phred qualities

    Returns
    -------
    np.ndarray | float
        List of error probabilities
    """
    if isinstance(q, float):
        return 10 ** (-q / 10)
    p = np.power(10, -np.array(q, dtype=float) / 10)
    return p


def unphred_str(strq: str) -> np.ndarray:
    """Converts string of qualities to array of error probabilities

    Parameters
    ----------
    strq : str
        BQ-like string

    Returns
    -------
    np.ndarray
        Array of error probabilities
    """
    q = [ord(x) - 33 for x in strq]
    return unphred(q)


def log_binomial(n: int, k: int, p: float) -> float:
    """Log-likelihood of observing k successes in n Bernoulli(p) trials.

    Probability is clamped to [1e-9, 1 - 1e-9] to avoid -inf at the boundaries.
    Returns 0.0 when n == 0 and k == 0; returns -inf for impossible (k, n) pairs
    (k < 0, k > n, or n == 0 with k != 0).
    """
    if k < 0 or k > n:
        return -math.inf
    if n == 0:
        return 0.0
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)
