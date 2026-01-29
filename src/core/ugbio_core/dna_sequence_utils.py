import re
from os.path import isfile

import numpy as np
import pyfaidx


def revcomp(seq: str | list | np.ndarray) -> str | list | np.ndarray:
    """Reverse complements DNA given as string

    Parameters
    ----------
    :param: seq Union[str,list,np.ndarray]
        DNA string
    :raises ValueError: is seq is not of the right type

    :return: str | list | np.ndarray


    """
    complement = {
        "A": "T",
        "C": "G",
        "G": "C",
        "T": "A",
        "a": "t",
        "c": "g",
        "g": "c",
        "t": "a",
    }
    if isinstance(seq, str):
        reverse_complement = "".join(complement.get(base, base) for base in reversed(seq))
    elif isinstance(seq, list):
        reverse_complement = [complement.get(base, base) for base in reversed(seq)]
    elif isinstance(seq, np.ndarray):
        reverse_complement = np.array([complement.get(base, base) for base in reversed(seq)])
    else:
        raise ValueError(f"Got unexpected variable {seq} of type {type(seq)}, expected str, list or numpy array")

    return reverse_complement


def hmer_length(seq: pyfaidx.Sequence, start_point: int) -> int:
    """Return length of hmer starting at point start_point

    Parameters
    ----------
    seq: pyfaidx.Sequence
        Sequence
    start_point: int
        Starting point

    Returns
    -------
    int
        Length of hmer (at least 1)
    """

    idx = start_point
    while seq[idx].seq.upper() == seq[start_point].seq.upper():
        idx += 1
    return idx - start_point


def get_chr_sizes(sizes_file: str) -> dict:
    """Returns dictionary from chromosome name to size

    Parameters
    ----------
    sizes_file: str
        .sizes file (use e.g.  cut -f1,2 Homo_sapiens_assembly19.fasta.fai > Homo_sapiens_
        assembly19.fasta.sizes to generate), .fai file or .dict file.
        Any file which doesnot end with fai or dict will be considered .sizes

    Returns
    -------
    dict:
        Dictionary from name to size
    """

    if not isfile(sizes_file):
        raise FileNotFoundError(f"Input_file {sizes_file} not found")
    if sizes_file.endswith("dict"):
        chrom_sizes = {}
        with open(sizes_file, encoding="utf-8") as file:
            for line in file:
                if line.startswith("@SQ"):
                    fields = line[3:].strip().split("\t")
                    row = {}  # Dict of all fields in the row
                    for field in fields:
                        key, value = field.split(":", 1)
                        row[key] = value
                    chrom_sizes[row["SN"]] = int(row["LN"])
        return chrom_sizes
    with open(sizes_file, encoding="ascii") as sizes:
        tmp = [x.strip().split()[:2] for x in sizes]
    return {x[0]: int(x[1]) for x in tmp}


def get_max_softclip_len(cigar):
    group = re.match("(?P<start>[0-9]+S)?[0-9]+[0-9MID]+[MID](?P<end>[0-9]+S)?", cigar).groups()
    start = int(group[0][:-1]) if group[0] else 0
    end = int(group[1][:-1]) if group[1] else 0
    return max(start, end)


# CIGAR operation codes per SAM specification
CIGAR_OPS = {
    "M": 0,  # Match or mismatch
    "I": 1,  # Insertion
    "D": 2,  # Deletion
    "N": 3,  # Skipped region (intron)
    "S": 4,  # Soft clip
    "H": 5,  # Hard clip
    "P": 6,  # Padding
    "=": 7,  # Sequence match
    "X": 8,  # Sequence mismatch
}


def parse_cigar_string(cigar_str: str) -> list[tuple[int, int]]:
    """
    Parse CIGAR string into list of (operation, length) tuples.

    This function parses CIGAR strings in the same format as pysam, converting
    CIGAR operations to their numeric codes according to the SAM specification.

    Parameters
    ----------
    cigar_str : str
        CIGAR string (e.g., "50M30S" or "30S50M")

    Returns
    -------
    list[tuple[int, int]]
        List of (operation, length) tuples where operation is a numeric code:
        M=0, I=1, D=2, N=3, S=4, H=5, P=6, ==7, X=8

    Examples
    --------
    >>> parse_cigar_string("50M30S")
    [(0, 50), (4, 30)]
    >>> parse_cigar_string("30S50M")
    [(4, 30), (0, 50)]
    """
    operations = []
    i = 0

    while i < len(cigar_str):
        # Parse number
        num_str = ""
        while i < len(cigar_str) and cigar_str[i].isdigit():
            num_str += cigar_str[i]
            i += 1
        if num_str and i < len(cigar_str):
            length = int(num_str)
            op_char = cigar_str[i]
            op_code = CIGAR_OPS.get(op_char, -1)
            if op_code != -1:
                operations.append((op_code, length))
            i += 1

    return operations


# CIGAR operations that consume reference bases (per SAM specification)
CIGAR_CONSUMES_REFERENCE = {
    CIGAR_OPS["M"],  # Match or mismatch
    CIGAR_OPS["D"],  # Deletion
    CIGAR_OPS["N"],  # Skipped region (intron)
    CIGAR_OPS["="],  # Sequence match
    CIGAR_OPS["X"],  # Sequence mismatch
}


def get_reference_alignment_end(reference_start: int, cigar: str) -> int:
    """
    Calculate the reference alignment end position from start and CIGAR string.

    The end position is calculated by summing the lengths of CIGAR operations
    that consume reference bases (M, D, N, =, X) and adding to the start position.
    The returned end position is exclusive (one past the last aligned base),
    following Python's half-open interval convention.

    Parameters
    ----------
    reference_start : int
        0-based reference start position of the alignment
    cigar : str
        CIGAR string (e.g., "50M2D30M" or "30S50M10I20M")

    Returns
    -------
    int
        0-based exclusive end position on the reference

    Examples
    --------
    >>> get_reference_alignment_end(100, "50M")
    150
    >>> get_reference_alignment_end(100, "30S50M")
    150
    >>> get_reference_alignment_end(100, "50M2D30M")
    182
    >>> get_reference_alignment_end(100, "50M10I30M")
    180
    """
    cigar_tuples = parse_cigar_string(cigar)
    reference_consumed = sum(length for op_code, length in cigar_tuples if op_code in CIGAR_CONSUMES_REFERENCE)
    return reference_start + reference_consumed
