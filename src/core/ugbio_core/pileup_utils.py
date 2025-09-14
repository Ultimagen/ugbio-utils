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
    chrom, pos, bases = fields[0], fields[1], fields[4]
    ref_count, nonref_count = parse_bases(bases)
    return chrom, pos, ref_count, nonref_count
