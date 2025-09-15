#!/env/python
# Copyright 2023 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# DESCRIPTION
#    integrate mpileup information into a somatic featuremap pileup VCF file
# CHANGELOG in reverse chronological order

import argparse
import logging
import os
import subprocess
import sys
from collections import deque
from os.path import join as pjoin

import pysam
from ugbio_core import misc_utils, pileup_utils
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
    header.formats.add(
        f"ref_counts_pm_{distance_start_to_center}",
        ".",
        "Integer",
        f"Reference counts at positions ±0..{distance_start_to_center}",
    )
    header.formats.add(
        f"nonref_counts_pm_{distance_start_to_center}",
        ".",
        "Integer",
        f"Non-reference counts at positions ±0..{distance_start_to_center}",
    )

    return header


def process_padded_positions(
    new_record: pysam.VariantRecord,
    distance_start_to_center: int,
    center_position: int,
    tumor_chunk_info: deque,
    normal_chunk_info: deque,
) -> tuple[pysam.VariantRecord, bool]:
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
    tumor_chunk_info : deque
        Dictionary containing tumor sample information where keys are indices and
        values are tuples/lists with:
        - index 0: chromosome (str)
        - index 1: actual genomic position (int)
        - index 2: reference read count (int)
        - index 3: non-reference read count (int)
    normal_chunk_info : deque
        Dictionary containing normal sample information with the same structure as tumor_chunk_info.

    Returns
    -------
    pysam.VariantRecord
        The updated variant record with position-specific reference and non-reference counts
        added to both tumor (sample 0) and normal (sample 1) samples. Field names follow
        the pattern 'ref_m{N}' or 'nonref_m{N}' for negative distances and 'ref_{N}' or
        'nonref_{N}' for non-negative distances.
    bool
        A boolean flag indicating whether any position missing within the specified range

    Notes
    -----
    The function assumes that tumor_chunk_info and normal_chunk_info have sequential
    integer keys starting from 0, with each entry containing at least 4 elements where
    index 2 is the reference count and index 3 is the non-reference count.
    """

    window_size = 2 * distance_start_to_center + 1
    # Create lists to store ref and nonref counts ordered by position
    tumor_ref_counts = [None] * window_size
    tumor_nonref_counts = [None] * window_size
    normal_ref_counts = [None] * window_size
    normal_nonref_counts = [None] * window_size

    # Build position maps
    for pileup_info in tumor_chunk_info:
        actual_position = int(pileup_info[1])
        dist = actual_position - center_position
        if abs(dist) <= distance_start_to_center:
            tumor_ref_counts[dist + distance_start_to_center] = pileup_info[2]
            tumor_nonref_counts[dist + distance_start_to_center] = pileup_info[3]

    for pileup_info in normal_chunk_info:
        actual_position = int(pileup_info[1])
        dist = actual_position - center_position
        if abs(dist) <= distance_start_to_center:
            normal_ref_counts[dist + distance_start_to_center] = pileup_info[2]
            normal_nonref_counts[dist + distance_start_to_center] = pileup_info[3]

    # Assign lists to the new fields
    new_record.samples[0][f"ref_counts_pm_{distance_start_to_center}"] = tumor_ref_counts
    new_record.samples[0][f"nonref_counts_pm_{distance_start_to_center}"] = tumor_nonref_counts
    new_record.samples[1][f"ref_counts_pm_{distance_start_to_center}"] = normal_ref_counts
    new_record.samples[1][f"nonref_counts_pm_{distance_start_to_center}"] = normal_nonref_counts
    has_missing = False
    if None in sum([tumor_ref_counts, tumor_nonref_counts, normal_ref_counts, normal_nonref_counts], []):
        has_missing = True
    return new_record, has_missing


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
    window_size = 2 * args.distance_start_to_center + 1
    # Iterators
    with open(args.tumor_mpileup) as f1, open(args.normal_mpileup) as f2:
        it1 = misc_utils.BufferedFileIterator(f1, window_size, pileup_utils.parse_mpileup_line)
        it2 = misc_utils.BufferedFileIterator(f2, window_size, pileup_utils.parse_mpileup_line)
        p1, p2 = next(it1, None), next(it2, None)

        # Open output VCF
        with pysam.VariantFile(out_sfmp_vcf, "w", header=header) as vcf_out:
            current_chrom = None

            for record in main_vcf.fetch():
                chrom = record.chrom
                start = record.pos - args.distance_start_to_center

                # --- handle chromosome change ---
                if chrom != current_chrom:
                    current_chrom = chrom

                    # advance pileup1 until correct chromosome
                    while p1 and (it1.buffer[0][0] != chrom or len(it1.buffer) < window_size):
                        p1 = next(it1, None)

                    # advance pileup2 until correct chromosome
                    while p2 and (it2.buffer[0][0] != chrom or len(it1.buffer) < window_size):
                        p2 = next(it2, None)

                # --- pileup1 ---
                while p1 and it1.buffer[0][0] == chrom and it1.buffer[0][1] < start:
                    p1 = next(it1, None)
                while p2 and it2.buffer[0][0] == chrom and it2.buffer[0][1] < start:
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

                # copy INFO fields
                for k, v in record.info.items():
                    if k in header.info:
                        new_record.info[k] = v

                # copy FORMAT fields
                for sample in record.samples:
                    src = record.samples[sample]
                    tgt = new_record.samples[sample]
                    for k, v in src.items():
                        if v in (None, (None,)):
                            continue  # no need to assign missing values
                        tgt[k] = v

                # process mpileup buffers
                new_record, is_missing = process_padded_positions(
                    new_record, args.distance_start_to_center, record.pos, it1.buffer, it2.buffer
                )

                # filter flag if missing positions
                if not is_missing:
                    new_record.filter.clear()
                    new_record.filter.add("PASS")
                else:
                    new_record.filter.add("MpileupData")

                vcf_out.write(new_record)

    logger.info(f"Merged VCF file with mpileup info created: {out_sfmp_vcf}")

    # Index the filtered VCF
    cmd_index = ["bcftools", "index", "-t", out_sfmp_vcf]
    logger.debug(" ".join(cmd_index))
    subprocess.run(cmd_index, check=True)


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
