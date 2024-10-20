import numpy as np


def searchsorted2d(ar_a: np.ndarray, ar_b: np.ndarray) -> np.ndarray:
    """
    Inserts ith element of b into sorted ith row of a

    Parameters
    ----------
    ar_a: np.ndarray
            rxc matrix, each rows is sorted
    ar_b: np.ndarray
            rx1 vector

    Returns
    -------
    np.ndarray
            rx1 vector of locations
    """
    dim1_a, dim2_a = ar_a.shape
    ar_b = ar_b.ravel()
    if ar_b.shape[0] != ar_a.shape[0]:
        raise ValueError("Number of values of array b must equal number of rows of array a")
    max_num = np.maximum(ar_a.max() - ar_a.min(), ar_b.max() - ar_b.min()) + 1
    r_seq = max_num * np.arange(ar_a.shape[0])
    indices = np.searchsorted(((ar_a.T + r_seq).T).ravel(), ar_b + r_seq)
    return indices - dim2_a * np.arange(dim1_a)


def shiftarray(arr: np.ndarray, num: int, fill_value: np.float64 = np.nan) -> np.ndarray:
    """Shifts array by num to the right

    Parameters
    ----------
    arr: np.ndarray
        Array to be shifted
    num: int
        Shift size (negative - left shift)
    fill_value: np.float64
        Fill value
    """
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def idx_last_nz(inp: np.ndarray | list) -> np.ndarray:
    """Index of the closest previous nonzero element for each element in the array.
    If the array starts with 0 - the index is -1

    Parameters
    ----------
    inp : np.ndarray
        Input array

    Returns
    -------
    np.ndarray
    """
    if not isinstance(inp, np.ndarray):
        inp = np.array(inp)
    nzs = np.concatenate(([-1], np.nonzero(inp)[0]))
    nzcounts = np.cumsum(inp > 0)
    return nzs[nzcounts]


def idx_next_nz(inp: np.ndarray | list) -> np.ndarray:
    """Index of the closest next nonzero element for each element in the array.
    If the array starts with 0 - the index is len(input)

    Parameters
    ----------
    inp : np.ndarray
        Input array

    Returns
    -------
    np.ndarray
    """
    result = idx_last_nz(inp[::-1])
    result = len(inp) - result - 1
    return result[::-1]
