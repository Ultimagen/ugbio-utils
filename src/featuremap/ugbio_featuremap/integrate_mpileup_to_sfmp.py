import argparse
import logging
import os
import subprocess
import sys
from collections import deque
from os.path import join as pjoin
from typing import Any

import numpy as np
import pysam
from ugbio_core.logger import logger


def __parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse command line arguments for mpileup info integration.

    Parameters
    ----------
    argv : list[str]
        Command line arguments, typically sys.argv

    Returns
    -------
    argparse.Namespace
        Parsed command line arguments containing:
        - sfmp_vcf : str
            Path to somatic featuremap pileup VCF file with tumor as sample[0] and normal as sample[1]
        - tumor_mpileup : str
            Path to tumor mpileup file
        - normal_mpileup : str
            Path to normal mpileup file
        - distance_start_to_center : int
            Distance from variant to adjacent positions to include in mpileup
        - out_directory : str
            Output directory for intermediate and final files (default: '.')
    """
    parser = argparse.ArgumentParser(
        prog="integrate_mpileup_to_sfmp.py",
        description=run.__doc__,
    )
    parser.add_argument(
        "--sfmp_vcf",
        help="somatic featuremap pileup vcf with tumor as sample[0] and normal as sample[1]",
        required=True,
        type=str,
    )
    parser.add_argument("--tumor_mpileup", help="tumor mpileup file", required=True, type=str)
    parser.add_argument("--normal_mpileup", help="normal mpileup file", required=True, type=str)
    parser.add_argument(
        "--distance_start_to_center",
        help="Distance from variant to adjacent positions to include in mpileup",
        required=True,
        type=int,
    )
    parser.add_argument(
        "--out_directory",
        help="out directory where intermediate and output files will be saved."
        " if not supplied all files will be written to current directory",
        required=False,
        type=str,
        default=".",
    )
    return parser.parse_args(argv[1:])


def create_new_header(main_vcf: pysam.VariantFile, distance_start_to_center: int) -> pysam.VariantHeader:
    """
    Create a new VCF header with additional format fields for reference and non-reference counts.

    Parameters
    ----------
    main_vcf : pysam.VariantFile
        The main VCF file whose header will be copied and extended.
    distance_start_to_center : int
        Maximum distance from center position for which format fields will be created.

    Returns
    -------
    pysam.VariantHeader
        A copy of the original header with additional format fields for reference and
        non-reference counts at each position from 0 to distance_start_to_center.

    Notes
    -----
    For position 0, only ref_0 and nonref_0 fields are added.
    For positions > 0, four fields are added: ref_N, nonref_N, ref_mN, nonref_mN,
    where N is the position and mN represents the negative position.
    """
    header = main_vcf.header.copy()

    # add new filter for insufficient mpileup data
    header.filters.add(
        "MpileupData",  # ID
        None,  # Number (None for FILTER)
        None,  # Type (None for FILTER)
        "Variants with partial/no mpileup information",  # Description
    )

    for loc in np.arange(0, distance_start_to_center + 1, 1):
        if loc == 0:
            fmt_fields = [f"ref_{loc}", f"nonref_{loc}"]
            fmt_description = [f"Reference count at {loc}", f"Non-reference count at {loc}"]
        else:
            fmt_fields = [f"ref_{loc}", f"nonref_{loc}", f"ref_m{loc}", f"nonref_m{loc}"]
            fmt_description = [
                f"Reference count at {loc}",
                f"Non-reference count at {loc}",
                f"Reference count at -{loc}",
                f"Non-reference count at -{loc}",
            ]
        for fmt, desc in zip(fmt_fields, fmt_description, strict=False):
            header.formats.add(
                fmt,
                1,
                "Integer",
                desc,
            )
    return header


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


def process_padded_positions(
    new_record: pysam.VariantRecord,
    distance_start_to_center: int,
    center_position: int,
    tumor_chunk_info: dict,
    normal_chunk_info: dict,
):
    """
    This function iterates through positions in tumor and normal chunk information,
    calculating distances from a center position and updating the variant record
    samples with reference and non-reference read counts at each position.
    Only positions within [-distance_start_to_center, +distance_start_to_center]
    will be processed.

    Parameters
    ----------
    new_record : pysam.VariantRecord
        The variant record to be updated with position information.
    distance_start_to_center : int
        The maximum distance from the center position to process in both directions.
        Positions will range from -distance_start_to_center to +distance_start_to_center.
    tumor_chunk_info : dict
        Dictionary containing tumor sample information where keys are indices and
        values are tuples/lists with:
        - index 0: chromosome (str)
        - index 1: actual genomic position (int)
        - index 2: reference read count (int)
        - index 3: non-reference read count (int)
    normal_chunk_info : dict
        Dictionary containing normal sample information with the same structure as tumor_chunk_info.

    Returns
    -------
    pysam.VariantRecord
        The updated variant record with position-specific reference and non-reference counts
        added to both tumor (sample 0) and normal (sample 1) samples. Field names follow
        the pattern 'ref_m{N}' or 'nonref_m{N}' for negative distances and 'ref_{N}' or
        'nonref_{N}' for non-negative distances.

    Notes
    -----
    The function assumes that tumor_chunk_info and normal_chunk_info have sequential
    integer keys starting from 0, with each entry containing at least 4 elements where
    index 2 is the reference count and index 3 is the non-reference count.
    """
    # Get the actual positions from the chunk info to calculate distances
    for pileup_info in tumor_chunk_info:
        # Calculate distance from center position
        actual_position = int(pileup_info[1])
        dist = actual_position - center_position

        # Format the key based on distance
        if (dist < 0) and (dist >= -distance_start_to_center):
            new_key = f"m{abs(dist)}"
        elif (dist >= 0) and (dist <= distance_start_to_center):
            new_key = f"{dist}"

        # Only update if within the expected distance range
        if abs(dist) <= distance_start_to_center:
            new_record.samples[0][f"ref_{new_key}"] = pileup_info[2]
            new_record.samples[0][f"nonref_{new_key}"] = pileup_info[3]

    for pileup_info in normal_chunk_info:
        # Calculate distance from center position
        actual_position = int(pileup_info[1])
        dist = actual_position - center_position

        # Format the key based on distance
        if (dist < 0) and (dist >= -distance_start_to_center):
            new_key = f"m{abs(dist)}"
        elif (dist >= 0) and (dist <= distance_start_to_center):
            new_key = f"{dist}"

        # Only update if within the expected distance range
        if abs(dist) <= distance_start_to_center:
            new_record.samples[1][f"ref_{new_key}"] = pileup_info[2]
            new_record.samples[1][f"nonref_{new_key}"] = pileup_info[3]

    return new_record


def _n_genotypes(n_alleles: int, ploidy: int) -> int:
    """
    Calculate the number of possible genotypes given the number of alleles and ploidy.

    For diploid organisms (ploidy=2), uses a shortcut formula. For other ploidy levels,
    computes the number of combinations with repetition (multichoose).

    Parameters
    ----------
    n_alleles : int
        The number of distinct alleles at the locus.
    ploidy : int
        The ploidy level (number of sets of chromosomes).

    Returns
    -------
    int
        The number of possible genotypes.

    Examples
    --------
    >>> _n_genotypes(2, 2)
    3
    >>> _n_genotypes(3, 2)
    6
    >>> _n_genotypes(2, 3)
    4
    """
    # combinations with repetition: C(n_alleles + ploidy - 1, ploidy)
    # diploid shortcut: n*(n+1)//2
    diploid_ploidy = 2
    if ploidy == diploid_ploidy:
        return n_alleles * (n_alleles + 1) // 2
    # generic
    num = 1
    den = 1
    k = ploidy
    n = n_alleles + ploidy - 1
    for i in range(1, k + 1):
        num *= n - (i - 1)
        den *= i
    return num // den


def _as_tuple(x: list | tuple | Any) -> tuple:
    """
    Convert input to a tuple.

    Parameters
    ----------
    x : list, tuple, or any type
        The input to be converted to a tuple. If `x` is a list or tuple, it is converted to a tuple.
        Otherwise, `x` is wrapped in a single-element tuple.

    Returns
    -------
    tuple
        The input as a tuple.
    """
    return tuple(x) if isinstance(x, list | tuple) else (x,)


def copy_format_fields_between_pysam_records(  # noqa: C901, PLR0912, PLR0915
    record: pysam.VariantRecord,
    new_record: pysam.VariantRecord,
    header: pysam.VariantHeader,
):
    """
    Copy all FORMAT fields for overlapping samples from a source pysam VariantRecord to a target VariantRecord.

    This function transfers all FORMAT fields (sample-level data) from `record` (source) to `new_record` (target)
    for samples present in both records. It ensures that the FORMAT fields are compatible with the target record's
    header and allele structure, handling allele-dependent fields (Number=A/R/G) and ploidy appropriately.

    Parameters
    ----------
    record : pysam.VariantRecord
        The source VariantRecord from which to copy FORMAT fields.
    new_record : pysam.VariantRecord
        The target VariantRecord to which FORMAT fields will be copied.
    header : pysam.VariantHeader
        The header of the target VCF, used to check FORMAT field definitions.

    Notes
    -----
    - Only samples present in both `record` and `new_record` are processed.
    - FORMAT fields not declared in the target header are skipped.
    - For allele-dependent fields (Number=A/R/G), the function checks that the value lengths
      match the target record's alleles.
    - Fields with variable or fixed length >1 are passed as tuples; scalars are wrapped as needed.
    - If a FORMAT field's value is incompatible with the target record (e.g., wrong length),
      it is skipped for that sample.
    """
    nalt_tgt = len(new_record.alts or ())

    for sample in new_record.samples:
        if sample not in record.samples:
            continue

        # Determine ploidy from GT in the *source* (fallback to target if missing)
        gt_src = record.samples[sample].get("GT", None)
        gt_tgt = new_record.samples[sample].get("GT", None)
        ploidy = (
            len(gt_src)
            if isinstance(gt_src, list | tuple) and gt_src is not None
            else (len(gt_tgt) if isinstance(gt_tgt, list | tuple) and gt_tgt is not None else 2)
        )

        for fmt_field in record.format.keys():
            # Skip if not declared in target header
            fmt_def = header.formats.get(fmt_field)
            if fmt_def is None:
                continue

            # Do not try to write GT unless you really want to overwrite it.
            if fmt_field == "GT":
                val = record.samples[sample].get("GT", None)
                if val is None:
                    continue
                # pysam expects a tuple of ints/None, e.g. (0,1) or (None, None)
                if not isinstance(val, list | tuple):
                    # Single allele -> make tuple
                    val = (val,)
                new_record.samples[sample]["GT"] = tuple(val)
                continue

            # Get value from source; skip if unset
            if fmt_field not in record.samples[sample]:
                continue
            val = record.samples[sample][fmt_field]
            if val is None:
                continue

            number = fmt_def.number  # could be int, or one of 'A','R','G','.'
            # For Number=A/R/G we must ensure allele-dependent lengths match the *target* record
            if number == "A":
                expected = nalt_tgt
                seq = _as_tuple(val)
                if len(seq) != expected:
                    # If allele counts differ, safest is to skip to avoid invalid lengths
                    continue
                new_record.samples[sample][fmt_field] = tuple(seq)
            elif number == "R":
                expected = 1 + nalt_tgt
                seq = _as_tuple(val)
                if len(seq) != expected:
                    continue
                new_record.samples[sample][fmt_field] = tuple(seq)
            elif number == "G":
                n_alleles_tgt = 1 + nalt_tgt
                expected = _n_genotypes(n_alleles_tgt, ploidy)
                seq = _as_tuple(val)
                if len(seq) != expected:
                    continue
                new_record.samples[sample][fmt_field] = tuple(seq)
            elif number == 1:
                # Single value expected: if a tuple/list was provided, take first element
                if isinstance(val, list | tuple):
                    new_record.samples[sample][fmt_field] = val[0]
                else:
                    new_record.samples[sample][fmt_field] = val
            elif number == "." or (isinstance(number, int) and number > 1):
                # Variable or fixed-length >1: pass tuples as-is, scalars get wrapped once
                if isinstance(val, list | tuple):
                    new_record.samples[sample][fmt_field] = tuple(val)
                else:
                    new_record.samples[sample][fmt_field] = (val,)
            else:
                # Fallback: try best-effort without re-shaping
                new_record.samples[sample][fmt_field] = val


def run(argv):  # noqa: C901, PLR0912, PLR0915
    """
    Integrate mpileup data into SFMP VCF file.

    This function reads a SFMP VCF file and two mpileup files (tumor and normal),
    then creates a new VCF file with integrated mpileup information for positions
    within a specified window around each variant.

    Parameters
    ----------
    argv : list of str
        Command line arguments containing:
        - sfmp_vcf : Path to input SFMP VCF file
        - tumor_mpileup : Path to tumor mpileup file
        - normal_mpileup : Path to normal mpileup file
        - out_directory : Output directory path
        - distance_start_to_center : Window size from variant position

    Returns
    -------
    None

    Notes
    -----
    The function creates a new VCF file with suffix '.mpileup.vcf.gz' in the
    specified output directory. It uses a sliding window approach to efficiently
    process mpileup data around each variant position. Only variants with complete
    mpileup data coverage (full window) are written to the output file.

    The output VCF file is automatically indexed using bcftools after creation.
    """
    args = __parse_args(argv)
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        handler.setLevel(logging.DEBUG)

    logger.info(f"Output directory: {args.out_directory}")

    # Create output directory if it doesn't exist
    if not os.path.exists(args.out_directory):
        os.makedirs(args.out_directory)
        logger.info(f"Created output directory: {args.out_directory}")

    sfmp_vcf = args.sfmp_vcf
    out_sfmp_vcf = pjoin(args.out_directory, os.path.basename(sfmp_vcf).replace(".vcf.gz", ".mpileup.vcf.gz"))

    # Open input VCFs
    main_vcf = pysam.VariantFile(sfmp_vcf)
    header = create_new_header(main_vcf, args.distance_start_to_center)

    # Iterators
    f1, f2 = open(args.tumor_mpileup), open(args.normal_mpileup)
    it1, it2 = map(parse_mpileup_line, f1), map(parse_mpileup_line, f2)
    buf1, buf2 = deque(), deque()  # buffers for sliding window
    p1, p2 = next(it1, None), next(it2, None)

    # Open output VCF
    with pysam.VariantFile(out_sfmp_vcf, "wz", header=header) as vcf_out:
        current_chrom = None

        for record in main_vcf.fetch():
            chrom = record.chrom
            start, end = record.pos - args.distance_start_to_center, record.pos + args.distance_start_to_center

            # --- handle chromosome change ---
            if chrom != current_chrom:
                current_chrom = chrom
                buf1.clear()
                buf2.clear()

                # advance pileup1 until correct chromosome
                while p1 and p1[0] != chrom:
                    p1 = next(it1, None)

                # advance pileup2 until correct chromosome
                while p2 and p2[0] != chrom:
                    p2 = next(it2, None)

            # --- pileup1 ---
            buf1 = deque([x for x in buf1 if int(x[1]) >= start])
            while p1 and p1[0] == chrom and int(p1[1]) < start:
                p1 = next(it1, None)
            while p1 and p1[0] == chrom and start <= int(p1[1]) <= end:
                buf1.append(p1)
                p1 = next(it1, None)

            # --- pileup2 ---
            buf2 = deque([x for x in buf2 if int(x[1]) >= start])
            while p2 and p2[0] == chrom and int(p2[1]) < start:
                p2 = next(it2, None)
            while p2 and p2[0] == chrom and start <= int(p2[1]) <= end:
                buf2.append(p2)
                p2 = next(it2, None)

            # --- create new VCF record ---
            new_record = vcf_out.new_record(
                contig=record.chrom,
                start=record.start,
                stop=record.stop,
                id=record.id,
                qual=record.qual,
                alleles=record.alleles,
                filter=record.filter.keys(),
            )

            # copy INFO
            for k, v in record.info.items():
                if k in header.info:
                    new_record.info[k] = v

            # copy FORMAT
            copy_format_fields_between_pysam_records(record, new_record, header)

            # process mpileup buffers
            new_record = process_padded_positions(new_record, args.distance_start_to_center, record.pos, buf1, buf2)

            # filter flag if missing positions
            if not (
                len(buf1) == (2 * args.distance_start_to_center + 1)
                and len(buf2) == (2 * args.distance_start_to_center + 1)
            ):
                new_record.filter.add("MpileupData")

            vcf_out.write(new_record)

    main_vcf.close()
    logger.info(f"Merged VCF file with mpileup info created: {out_sfmp_vcf}")

    # Index the filtered VCF
    cmd_index = ["bcftools", "index", "-t", out_sfmp_vcf]
    logger.debug(" ".join(cmd_index))
    subprocess.check_call(cmd_index)


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
