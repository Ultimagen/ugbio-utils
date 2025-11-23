from __future__ import annotations

import itertools
import os
import os.path
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
import pysam


class BufferedFileIterator:
    """
    Iterator that reads lines from a file and maintains a buffer of the last window_size elements.

    This iterator wraps a file object and optionally applies a parsing function to each line,
    while maintaining a circular buffer of the most recent parsed elements.
    """

    def __init__(self, file_obj, window_size: int, parse_func: Callable[[str], Any] | None = None):
        """
        Initialize the buffered iterator.

        Parameters
        ----------
        file_obj : file-like object
            File object to read lines from
        window_size : int
            Size of the buffer to maintain (number of last elements to keep)
        parse_func : callable, optional
            Function to apply to each line. If None, returns the raw line (stripped).
            Function should take a string (line) and return the parsed result.
        """
        self.file_obj = file_obj
        self.window_size = window_size
        self.parse_func = parse_func
        self.buffer = deque(maxlen=window_size)
        self._current = None

    def __iter__(self):
        return self

    def __next__(self):
        """
        Get the next line (optionally parsed) and update the buffer.

        Returns
        -------
        Any
            Parsed line if parse_func is provided, otherwise stripped raw line

        Raises
        ------
        StopIteration
            When no more lines are available
        """
        line = next(self.file_obj, None)
        if line is None:
            raise StopIteration

        if self.parse_func is not None:
            parsed = self.parse_func(line)
        else:
            parsed = line.strip()

        self.buffer.append(parsed)
        self._current = parsed
        return parsed

    def get_buffer(self) -> deque:
        """
        Get the current buffer.

        Returns
        -------
        deque
            Copy of the current buffer containing the last window_size elements
        """
        return self.buffer

    def clear_buffer(self):
        """Clear the internal buffer."""
        self.buffer.clear()

    @property
    def current(self):
        """Get the current (most recent) parsed element."""
        return self._current

    def __len__(self):
        """Get the current buffer size."""
        return len(self.buffer)


def runs_of_one(array, axis=None):
    """
    returns start and end (half open) of intervals of ones in a binary vector
    if axis=None - runs are identified according to the (only) non-singleton axis
    """
    array = array.astype(np.int8)
    if isinstance(array, np.ndarray):
        array = np.array(array)
    if not axis:
        shapes = [x for x in array.shape if x != 1]
        if len(shapes) != 1:
            raise RuntimeError("runs_of_one - too many non-singleton axes in array")
        array = np.squeeze(array).reshape(1, -1)
        axis = 1
    if axis != 1:
        array.reshape(array.shape[::-1])
    runs_of_ones = []
    for i in range(array.shape[0]):
        one_line = array[i, :]

        diffs = np.diff(one_line)

        starts = np.nonzero(diffs == 1)[0] + 1
        if one_line[0] == 1:
            starts = np.concatenate(([0], starts))
        ends = np.nonzero(diffs == -1)[0] + 1
        if one_line[-1] == 1:
            ends = np.concatenate((ends, [len(one_line)]))

        runs_of_ones.append(zip(starts, ends, strict=False))

    return runs_of_ones


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
        raise RuntimeError("Number of values of array b equal number of rows of array a")
    max_num = np.maximum(ar_a.max() - ar_a.min(), ar_b.max() - ar_b.min()) + 1
    r_seq = max_num * np.arange(ar_a.shape[0])
    indices = np.searchsorted(((ar_a.T + r_seq).T).ravel(), ar_b + r_seq)
    return indices - dim2_a * np.arange(dim1_a)


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def shiftarray(arr: np.ndarray, num: int, fill_value: float = np.nan) -> np.ndarray:
    """Shifts array by num to the right

    Parameters
    ----------
    arr: np.ndarray
        Array to be shifted
    num: int
        Shift size (negative - left shift)
    fill_value: np.float
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


def contig_lens_from_bam_header(bam_file: str, output_file: str):
    """Creates a "sizes" file from contig lengths in bam header.
    Sizes file is per the UCSC spec: contig <tab> length

    Parameters
    ----------
    bam_file: str
        Bam file
    output_file: str
        Output file

    Returns
    -------
    None, writes output_file
    """

    with pysam.AlignmentFile(bam_file) as infile:
        with open(output_file, "w", encoding="ascii") as outfile:
            lengths = infile.header.lengths
            contigs = infile.header.references
            for contig, length in zip(contigs, lengths, strict=False):
                outfile.write(f"{contig}\t{length}\n")


def max_merits(specificity, recall):
    """Finds ROC envelope from multiple sets of specificity and recall"""
    n_rows = specificity.shape[0]
    ind_max = np.ones(n_rows, bool)
    for j in range(n_rows):
        for i in range(n_rows):
            if (specificity[i] > specificity[j]) & (recall[i] > recall[j]):
                ind_max[j] = False
                continue
    ind = np.where(ind_max)[0]
    ind_sort_recall = np.argsort(recall[ind])
    return ind[ind_sort_recall]


def is_pos_in_interval(pos: int, interval: tuple) -> bool:
    """Is position inside the [interval)

    Parameters
    ----------
    pos: int
        Position
    interval: tuple
        [start,end)

    Returns
    -------
    bool
    """
    return interval[0] <= pos < interval[1]


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


def cleanup_temp_files(temp_files: list[str]) -> None:
    """
    Remove temporary files and their indices.

    Parameters
    ----------
    temp_files : list[str]
        List of temporary file paths to remove
    """
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        # Also remove index files
        for ext in [".tbi", ".csi", ".idx", ".crai", ".bai"]:
            index_file = temp_file + ext
            if os.path.exists(index_file):
                os.unlink(index_file)
