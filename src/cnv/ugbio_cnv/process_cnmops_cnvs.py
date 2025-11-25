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


def get_parser():
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

    return parser


def merge_filter_files(original_bed: str, filter_files: list[str], output_file: str):
    """Merges multiple filter bed files into a single sorted bed file.

    Parameters
    ----------
    original_bed : str
        Original bed: four columns (chr, start, end, copy-number)
    filter_files : list[str]
        BED files with filters added to merge. Each file should have four columns (chr, start, end, CN2|LEN etc.)
        Some lines are empty
    output_file : str
        Output file: filters will be combined by ;, if all filters are empty, the result will be just CN
    """
    original_bed_df = pd.read_csv(original_bed, sep="\t", header=None, names=["chr", "start", "end", "copy-number"])
    original_bed_df = original_bed_df.set_index(["chr", "start", "end"])
    filter_dfs = []
    for filter_file in filter_files:
        filter_df = pd.read_csv(filter_file, sep="\t", header=None, names=["chr", "start", "end", "filter"])
        filter_df = filter_df.set_index(["chr", "start", "end"])
        filter_dfs.append(filter_df)
    merged_df = pd.concat((original_bed_df, *filter_dfs), axis=1, join="outer").fillna("")
    cols = ["copy-number"] + [f"filter_{i}" for i in range(len(filter_dfs))]
    merged_df.columns = cols
    merged_df["combine_filters"] = merged_df[cols[1:]].apply(lambda x: ";".join([y for y in x if y]), axis=1)
    merged_df["combined_cn"] = merged_df.apply(
        lambda x: x["copy-number"] if x["combine_filters"] == "" else x["combine_filters"], axis=1
    )
    merged_df = merged_df.reset_index()
    merged_df.to_csv(output_file, sep="\t", header=False, index=False, columns=["chr", "start", "end", "combined_cn"])


def annotate_bed(bed_file, lcr_cutoff, lcr_file, prefix, length_cutoff=10000):
    # get filters regions
    filter_files = []
    bu = bed_utils.BedUtils()

    if lcr_file is not None:
        lcr_bed_file = bu.filter_by_bed_file(bed_file, lcr_cutoff, lcr_file, prefix, "UG-CNV-LCR")
        filter_files.append(lcr_bed_file)

    if length_cutoff is not None and length_cutoff > 0:
        length_bed_file = bu.filter_by_length(bed_file, length_cutoff, prefix)
        filter_files.append(length_bed_file)

    if not filter_files:
        # No filters to apply, just return sorted bed file
        out_bed_file_sorted = prefix + os.path.splitext(os.path.basename(bed_file))[0] + ".annotate.bed"
        bu.bedtools_sort(bed_file, out_bed_file_sorted)
        return out_bed_file_sorted

    out_combined_info = prefix + os.path.splitext(os.path.basename(bed_file))[0] + ".unsorted.annotate.combined.bed"
    merge_filter_files(bed_file, filter_files, out_combined_info)
    # merge all filters and sort

    out_annotate = prefix + os.path.splitext(os.path.basename(bed_file))[0] + ".annotate.bed"
    bu.bedtools_sort(out_combined_info, out_annotate)
    os.unlink(out_combined_info)
    for f in filter_files:
        os.unlink(f)

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
        cov_df.columns = ["chrom", "start", "end", "name", "cov"]

        # Sort to match the primary bed file (first 3 columns are chr, start, end)
        cov_df = cov_df.sort_values(by=["chrom", "start", "end"]).reset_index(drop=True)

        # Extract the last column (the coverage value)
        coverage_values = cov_df["cov"]

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
    input_sample = ["sample", "cohort"]
    output_param = ["mean", "stdev"]

    for isamp in input_sample:
        for oparam in output_param:
            out_annotate_bed_file_cov = annotated_bed_file.replace(".annotate.bed", f".annotate.{isamp}.{oparam}.bed")
            input_cov_file = sample_norm_coverage_file if isamp == "sample" else cohort_avg_coverage_file
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
    Given a cn.mops bed file, this script will filter it by :
    1. lcr bed (ug_cnv_lcr) file
    3. length
    output is a VCF file with filtering tags
    """

    parser = get_parser()
    args = parser.parse_args(argv[1:])
    logger.setLevel(getattr(logging, args.verbosity))

    prefix = ""
    if args.out_directory:
        prefix = args.out_directory
        prefix = prefix.rstrip("/") + "/"

    # Annotate with lcr_file or min_cnv_length are provided, sort otherwise
    out_annotate_bed_file = annotate_bed(
        args.input_bed_file, args.intersection_cutoff, args.cnv_lcr_file, prefix, args.min_cnv_length
    )

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
