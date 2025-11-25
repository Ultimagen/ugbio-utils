# Copyright 2022 Ultima Genomics Inc.
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
#    Process CNV calls from cn.mops: filter by length and UG-CNV-LCR, annotate, and convert to VCF.
# CHANGELOG in reverse chronological order

import argparse
import logging
import os
import sys
import warnings

import pandas as pd
import ugbio_cnv.convert_combined_cnv_results_to_output_formats as vcf_writer
import ugbio_core.misc_utils as mu
from ugbio_core import bed_utils, vcf_utils
from ugbio_core.logger import logger

warnings.filterwarnings("ignore")

bedtools = "bedtools"
bedmap = "bedmap"


def annotate_bed(bed_file, lcr_cutoff, lcr_file, prefix, length_cutoff=10000):
    # get filters regions
    filter_files = []
    bed_utils_instance = bed_utils.BedUtils()

    if lcr_file is not None:
        lcr_bed_file = bed_utils_instance.filter_by_bed_file(bed_file, lcr_cutoff, lcr_file, prefix, "UG-CNV-LCR")
        filter_files.append(lcr_bed_file)

    if length_cutoff is not None and length_cutoff > 0:
        length_bed_file = bed_utils_instance.filter_by_length(bed_file, length_cutoff, prefix)
        filter_files.append(length_bed_file)

    if not filter_files:
        # No filters to apply, just return sorted bed file
        out_bed_file_sorted = prefix + os.path.splitext(os.path.basename(bed_file))[0] + ".sorted.annotate.bed"
        cmd = bedtools + " sort -i " + bed_file + " > " + out_bed_file_sorted
        os.system(cmd)  # noqa: S605
        logger.info(cmd)
        return out_bed_file_sorted

    # merge all filters and sort
    out_filters_unsorted = prefix + "filters.annotate.unsorted.bed"
    cmd = "cat " + " ".join(filter_files) + " > " + out_filters_unsorted

    os.system(cmd)  # noqa: S605
    logger.info(cmd)

    out_filters_sorted = prefix + "filters.annotate.bed"
    cmd = bedtools + " sort -i " + out_filters_unsorted + " > " + out_filters_sorted
    os.system(cmd)  # noqa: S605
    logger.info(cmd)

    out_bed_file_sorted = prefix + os.path.splitext(os.path.basename(bed_file))[0] + ".sorted.bed"
    cmd = bedtools + " sort -i " + bed_file + " > " + out_bed_file_sorted
    os.system(cmd)  # noqa: S605
    logger.info(cmd)

    # annotate bed files by filters
    out_unsorted_annotate = prefix + os.path.splitext(os.path.basename(bed_file))[0] + ".unsorted.annotate.bed"
    cmd = (
        bedmap
        + " --echo --echo-map-id-uniq --delim '\\t' "
        + out_bed_file_sorted
        + " "
        + out_filters_sorted
        + " > "
        + out_unsorted_annotate
    )
    os.system(cmd)  # noqa: S605
    logger.info(cmd)
    # combine to 4th column

    out_combined_info = prefix + os.path.splitext(os.path.basename(bed_file))[0] + ".unsorted.annotate.combined.bed"
    cmd = (
        "cat "
        + out_unsorted_annotate
        + ' | awk \'{if($5=="")'
        + "{print $0}"
        + 'else{print $1"\t"$2"\t"$3"\t"$5}}\' | sed \'s/\t$//\' > '
        + out_combined_info
    )
    os.system(cmd)  # noqa: S605
    logger.info(cmd)

    out_annotate = prefix + os.path.splitext(os.path.basename(bed_file))[0] + ".annotate.bed"
    cmd = "sort -k1,1V -k2,2n -k3,3n " + out_combined_info + " > " + out_annotate
    os.system(cmd)  # noqa: S605
    logger.info(cmd)

    return out_annotate


def aggregate_annotations_in_df(
    primary_bed_file: str, coverage_annotations: list[tuple[str, str, str]]
) -> pd.DataFrame:
    """
    Aggregate multiple annotation bed files into a single DataFrame.

    This function reads a primary bed file and merges coverage annotations from multiple
    bed files into a single DataFrame. The primary bed file's 4th column contains
    semicolon-separated annotations, each of which has a copy number and optional filters.
    The copy number is extracted as an integer, and filters are stored as a tuple.

    Parameters
    ----------
    primary_bed_file : str
        Path to the primary bed file with 4 columns (chr, start, end, annotation).
        The 4th column contains annotations like "CN2", "CN3|UG-CNV-LCR", or
        "CN1|LEN;CN1|UG-CNV-LCR" (semicolon-separated, CN is always the same).
    coverage_annotations : list of tuple
        List of tuples, each containing (sample_name, operation, bed_file_path).
        Example: [('cov', 'mean', 'file1.bed'), ('cov', 'std', 'file2.bed')]

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: chr, start, end, CopyNumber (int), FILTER (tuple),
        and additional columns for each coverage annotation (e.g., CNMOPS_COV_MEAN,
        CNMOPS_COV_STD, CNMOPS_COHORT_MEAN, CNMOPS_COHORT_STD).

    Notes
    -----
    - All bed files are assumed to have the same regions in the same order
    - The function performs sorting to ensure proper alignment
    - Coverage annotation column names are formatted as CNMOPS_{SAMPLE}_{OPERATION}
      in uppercase (e.g., CNMOPS_COV_MEAN)
    - CopyNumber is converted to an integer by removing the "CN" prefix
    - filter is a tuple of filter names (("PASS",) tuple when no filters present)
    """
    # Read the primary bed file (no header, 4 columns)
    cnv_df = pd.read_csv(primary_bed_file, sep="\t", header=None, names=["chr", "start", "end", "annotation"])

    # Sort the DataFrame to ensure consistent ordering
    cnv_df = cnv_df.sort_values(["chr", "start", "end"]).reset_index(drop=True)

    # Parse the annotation column
    # Format: "CN2", "CN3|UG-CNV-LCR", or "CN1|LEN;CN1|UG-CNV-LCR"
    def parse_annotation(annotation_str):
        # Split by semicolon to get all annotations
        parts = annotation_str.split(";")

        # Extract copy number from first part (remove "CN" prefix)
        first_part = parts[0].split("|")
        copy_number = float(first_part[0].replace("CN", ""))

        # Collect all filters from all parts
        filters = []
        for part in parts:
            part_items = part.split("|")
            # Skip the CN part (first item) and add any filters
            filters.extend(part_items[1:])

        # Return copy number and tuple of filters (PASS tuple if no filters)
        return copy_number, ",".join(filters) if filters else "PASS"

    neutral = 2
    cnv_df[["CopyNumber", "filter"]] = cnv_df["annotation"].apply(lambda x: pd.Series(parse_annotation(x)))
    cnv_df["SVTYPE"] = cnv_df["CopyNumber"].apply(
        lambda x: "DUP" if x > neutral else ("DEL" if x < neutral else "NEUTRAL")
    )
    cnv_df = cnv_df.drop(columns=["annotation"])

    # Process each coverage annotation file
    for sample_name, operation, bed_file_path in coverage_annotations:
        # Read the coverage annotation bed file
        cov_df = pd.read_csv(bed_file_path, sep="\t", header=None)

        # Assign column names for sorting
        cov_df.columns = [f"col_{i}" for i in range(len(cov_df.columns))]

        # Sort to match the primary bed file (first 3 columns are chr, start, end)
        cov_df = cov_df.sort_values(by=["col_0", "col_1", "col_2"]).reset_index(drop=True)

        # Extract the last column (the coverage value)
        last_col_name = cov_df.columns[-1]
        coverage_values = cov_df[last_col_name]

        # Create column name in the format CNMOPS_{SAMPLE}_{OPERATION} (uppercase)
        col_name = f"CNMOPS_{sample_name.upper()}_{operation.upper()}"

        # Add the coverage values as a new column
        cnv_df[col_name] = coverage_values.to_numpy()

    return cnv_df


def _aggregate_coverages(
    annotated_bed_file: str, sample_norm_coverage_file: str, cohort_avg_coverage_file: str, tempdir: str
) -> list[tuple[str, str, str]]:
    """
    Prepare coverage annotations for aggregation.
    Parameters
    ----------
    annotated_bed_file : str
        Path to the annotated bed file.
    sample_norm_coverage_file : str
        Path to the sample normalized coverage bed file.
    cohort_avg_coverage_file : str
        Path to the cohort average coverage bed file.
    tempdir : str
        Directory to store intermediate files.
    Returns
    -------
    list of tuple
        List of tuples containing (sample/cohort cvg type, operation, bed_file_path) for coverage annotations.
    """
    coverage_annotations = []
    # annotate with coverage info
    input_sample = ["cov", "cohort"]
    output_param = ["mean", "stdev"]

    for isamp in input_sample:
        for oparam in output_param:
            out_annotate_bed_file_cov = annotated_bed_file.replace(".annotate.bed", f".annotate.{isamp}.{oparam}.bed")
            input_cov_file = sample_norm_coverage_file if isamp == "cov" else cohort_avg_coverage_file
            bed_utils.BedUtils().bedtools_map(
                a_bed=annotated_bed_file,
                b_bed=input_cov_file,
                output_bed=out_annotate_bed_file_cov,
                operation=oparam,
                presort=True,
                tempdir_prefix=tempdir,
                column=5,
            )
            coverage_annotations.append((isamp, oparam, out_annotate_bed_file_cov))
    return coverage_annotations


def add_ids(cnmops_cnv_df: pd.DataFrame) -> pd.DataFrame:
    """Add IDs to the CNV DataFrame in the format cnmops_<svtype>_<count>.

    Parameters
    ----------
    cnmops_cnv_df : pd.DataFrame
        Input

    Returns
    -------
    pd.DataFrame
        Output, ID added
    """

    # Add IDs in the format cnmops_<svtype>_<count>
    svtype_counts = {}
    ids = []
    for _, row in cnmops_cnv_df.iterrows():
        svtype = row["SVTYPE"].lower()
        svtype_counts[svtype] = svtype_counts.get(svtype, 0) + 1
        ids.append(f"cnmops_{svtype}_{svtype_counts[svtype]}")
    cnmops_cnv_df["ID"] = ids
    return cnmops_cnv_df


def run(argv):
    """
    Given a bed file, this script will filter it by :
    1. lcr bed (ug_cnv_lcr) file
    3. length
    output consists of 2 files:
    - VCF file with filtering tags
    - annotated bed file with filtering tags
    """
    parser = argparse.ArgumentParser(
        prog="process_cnmops_cnvs.py",
        description="Process CNV calls from cn.mops: filter, annotate, and convert to VCF",
    )
    parser.add_argument("--input_bed_file", help="input bed file with .bed suffix", required=True, type=str)
    parser.add_argument(
        "--intersection_cutoff",
        help="intersection cutoff for bedtools subtract function",
        required=False,
        type=float,
        default=0.5,
    )
    parser.add_argument("--cnv_lcr_file", help="UG-CNV-LCR bed file", required=False, type=str)
    parser.add_argument("--min_cnv_length", required=False, type=int, default=10000)
    parser.add_argument(
        "--out_directory",
        help="out directory where intermediate and output files will be saved."
        " if not supplied all files will be written to current directory",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--sample_norm_coverage_file", help="sample normalized coverage file (BED)", required=False, type=str
    )
    parser.add_argument("--cohort_avg_coverage_file", help="Cohort average coverage (BED)", required=False, type=str)

    parser.add_argument(
        "--verbosity",
        help="Verbosity: ERROR, WARNING, INFO, DEBUG",
        required=False,
        default="INFO",
    )
    parser.add_argument(
        "--fasta_index_file",
        help="tab delimeted file holding reference genome chr ids with their lengths. (.fai file)",
        required=True,
        type=str,
    )
    parser.add_argument("--sample_name", help="sample name", required=True, type=str)

    args = parser.parse_args(argv[1:])
    logger.setLevel(getattr(logging, args.verbosity))

    prefix = ""
    if args.out_directory:
        prefix = args.out_directory
        prefix = prefix.rstrip("/") + "/"
    if "--" in args.input_bed_file:
        cmd = f"cp {args.input_bed_file} {args.input_bed_file.replace('--','-TMP-')}"
        os.system(cmd)  # noqa: S605
        args.input_bed_file = args.input_bed_file.replace("--", "-TMP-")

    # Annotate if lcr_file or min_cnv_length is provided
    if args.cnv_lcr_file or args.min_cnv_length:
        out_annotate_bed_file = annotate_bed(
            args.input_bed_file, args.intersection_cutoff, args.cnv_lcr_file, prefix, args.min_cnv_length
        )
    else:
        # If no filtering, just sort the input bed file
        out_bed_file_sorted = prefix + os.path.splitext(os.path.basename(args.input_bed_file))[0] + ".annotate.bed"
        cmd = bedtools + " sort -i " + args.input_bed_file + " > " + out_bed_file_sorted
        os.system(cmd)  # noqa: S605
        logger.info(cmd)
        out_annotate_bed_file = out_bed_file_sorted

    target_file = out_annotate_bed_file.replace("-TMP-", "--")
    if out_annotate_bed_file != target_file:
        cmd = f"mv {out_annotate_bed_file} {target_file}"
        os.system(cmd)  # noqa: S605
    out_annotate_bed_file = target_file

    coverage_annotations = []
    if args.sample_norm_coverage_file and args.cohort_avg_coverage_file:
        coverage_annotations = _aggregate_coverages(
            out_annotate_bed_file, args.sample_norm_coverage_file, args.cohort_avg_coverage_file, args.out_directory
        )

    cnmops_cnv_df = aggregate_annotations_in_df(out_annotate_bed_file, coverage_annotations)
    cnmops_cnv_df = add_ids(cnmops_cnv_df)

    out_vcf_file = out_annotate_bed_file.replace(".bed", ".vcf.gz")
    vcf_writer.write_cnv_vcf(out_vcf_file, cnmops_cnv_df, args.sample_name, args.fasta_index_file)
    vu = vcf_utils.VcfUtils()
    vu.index_vcf(out_vcf_file)
    logger.info("Cleaning temporary files...")
    mu.cleanup_temp_files([out_annotate_bed_file] + [cov[2] for cov in coverage_annotations])
    logger.info(f"output files: {out_vcf_file}")


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
