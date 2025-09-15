import fileinput
from collections import defaultdict

import pandas as pd

PILEUP_IGNORE = {",", ".", "*", "#", ">", "<"}
USE_STARTEND = True


def pileup_to_freq(reference: str, pileup: str) -> dict[str, int]:  # noqa: C901 #TODO: refactor. too complex
    """Counts A, C, G, T, In, Del occurence given a samtools pileup string"""

    pileup = pileup.upper()
    frequencies = defaultdict(int)
    is_indel = False
    indel_pos = 0
    indel_size = ""

    is_start = 0
    last_base = ""

    for k in pileup:
        if k in ("+", "-"):
            is_indel = True
            indel_pos = 0
            indel_size = ""
            indel = k
            continue

        if k == "$":
            frequencies[last_base] -= not USE_STARTEND
            continue

        # Handle first base in read; Indels would come after (^)(QUAL)(BASE) structure
        if k == "^":
            is_start = 1
            continue
        if is_start > 0:
            if is_start < 2:  # noqa: PLR2004
                is_start += 1
            elif is_start == 2:  # noqa: PLR2004
                frequencies[k] += USE_STARTEND
                is_start = 0
            continue

        if is_indel:
            is_digit = str.isdigit(k)
            if is_digit:
                indel_size += k
                continue
            indel_pos += 1
            indel += k
            if indel_pos == int(indel_size):
                frequencies[indel] += 1
                is_indel = False
                indel_pos = 0
            continue

        last_base = k
        frequencies[k] += 1

    frequencies[reference] += frequencies[","] + frequencies["."]
    frequencies["Deletion"] = frequencies["*"] + frequencies["#"]

    return {k: v for k, v in frequencies.items() if v > 0 and k not in PILEUP_IGNORE}


def create_frequencies_from_pileup(input_pileup_file) -> pd.DataFrame:
    """Create a pandas DataFrame summerizing samtools pileup file results"""
    df_frequencies = pd.DataFrame(columns=["Depth", "Chrom", "Pos", "Ref", "Sample", "Base", "Count"])
    data = []
    with fileinput.input(input_pileup_file, mode="r") as fin:
        for line in fin:
            sample = 0
            values = line.split("\t")
            chr_name = values[0]
            pos = values[1]
            ref = values[2]
            dp = values[3]
            # POSITION  = ';'.join([str(values[i]) for i in range(3)])
            # Go through all seq fields
            for i in range(4, len(values), 3):
                sample += 1
                # No reads aligned to this position
                if values[i - 1] == "0":
                    continue
                freq = pileup_to_freq(values[2], values[i])
                for base, count in freq.items():
                    data.append([dp, chr_name, pos, ref, sample, base, count])
    fin.close()
    df_frequencies = pd.DataFrame(data, columns=["Depth", "Chrom", "Pos", "Ref", "Sample", "Base", "Count"])
    return df_frequencies


def parse_bases(bases: str) -> tuple[int, int]:
    """
    Parse base calls from mpileup format and count reference and non-reference bases.
    Parameters
    ----------
    bases : str
        String containing base calls from mpileup format. Can include:
        - '.' or ',' for reference bases
        - 'ACGTNacgtn*' for non-reference bases
        - '^' followed by mapping quality for read start
        - '$' for read end
        - '+' or '-' followed by length digits and inserted/deleted sequence
    Returns
    -------
    ref_count : int
        Number of reference base calls (. or ,)
    nonref_count : int
        Number of non-reference base calls (substitutions and indels)
    Notes
    -----
    The function handles mpileup format special characters:
    - Skips mapping quality after '^'
    - Ignores '$' markers
    - Properly parses indel notation (+/-) with length specification
    """
    ref_count = 0
    nonref_count = 0
    i = 0
    while i < len(bases):
        c = bases[i]
        if c in ".,":
            ref_count += 1
        elif c in "ACGTNacgtn*":
            nonref_count += 1
        elif c == "^":
            i += 1
        elif c == "$":
            pass
        elif c in "+-":
            i += 1
            length = ""
            nonref_count += 1  # count the indel itself
            while i < len(bases) and bases[i].isdigit():
                length += bases[i]
                i += 1
            i += int(length) - 1
        i += 1
    return ref_count, nonref_count


def parse_mpileup_line(mpileup_line: str) -> tuple[str, int, int, int]:
    """
    Parse a single mpileup line and extract base count information.
    Parameters
    ----------
    mpileup_line : str
        A single line from an mpileup file containing chromosome, position,
        reference base, depth, and base string information.
    Returns
    -------
    tuple[str, int, int, int]
        A tuple containing:
        - chrom : str
            Chromosome identifier
        - pos : int
            Genomic position
        - ref_count : int
            Count of bases matching the reference
        - nonref_count : int
            Count of bases not matching the reference
    Notes
    -----
    The function expects mpileup lines with at least 5 tab-delimited fields:
    chromosome, position, reference base, depth, and bases string.
    """
    fields = mpileup_line.strip().split("\t")
    minimal_num_fields = 5
    if len(fields) < minimal_num_fields:
        raise ValueError(
            f"Invalid mpileup line: expected at least {minimal_num_fields} fields, got {len(fields)} : {mpileup_line}"
        )
    chrom, pos, bases = fields[0], int(fields[1]), fields[4]
    ref_count, nonref_count = parse_bases(bases)
    return chrom, pos, ref_count, nonref_count
