import argparse
import logging
import os
import subprocess
import sys
import tempfile
from os.path import join as pjoin

import pandas as pd
import pysam
import ugbio_core.misc_utils as mu
from pyfaidx import Fasta
from ugbio_cnv.combine_cnv_vcf_utils import (
    cnv_vcf_to_bed,
    combine_vcf_headers_for_cnv,
    merge_cnvs_in_vcf,
    update_vcf_contigs,
    write_vcf_records_with_source,
)
from ugbio_cnv.convert_combined_cnv_results_to_output_formats import FILTER_TAG_REGISTRY, INFO_TAG_REGISTRY
from ugbio_core.bed_utils import BedUtils
from ugbio_core.logger import logger
from ugbio_core.vcf_utils import VcfUtils


def run_cmd(cmd):
    logger.info(cmd)
    subprocess.run(cmd, shell=True, check=True)  # noqa: S602


def __parse_args_concat(parser: argparse.ArgumentParser) -> None:
    """Add arguments specific to the concat tool."""
    parser.add_argument("--cnmops_vcf", help="input VCF file holding cn.mops CNV calls", required=True, type=str)
    parser.add_argument("--cnvpytor_vcf", help="input VCF file holding cnvpytor CNV calls", required=True, type=str)
    parser.add_argument("--output_vcf", help="output combined VCF file", required=True, type=str)
    parser.add_argument("--fasta_index", help="fasta.fai file", required=True, type=str)
    parser.add_argument("--out_directory", help="output directory", required=False, type=str)


def __parse_args_dup_length_filter(parser: argparse.ArgumentParser) -> None:
    """Add arguments specific to the duplication length_filter tool."""
    parser.add_argument("--combined_calls", help="Input combined CNV calls VCF file", required=True, type=str)
    parser.add_argument(
        "--combined_calls_annotated",
        help="Output combined CNV calls VCF file with filtering annotation",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--filtered_length",
        help="Minimum duplication length to be considered valid",
        required=True,
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--distance_threshold",
        help="Distance threshold for merging CNV segments",
        required=False,
        type=int,
        default=1500,
    )


def __parse_args_gaps_perc(parser: argparse.ArgumentParser) -> None:
    """Add arguments specific to the gaps percentage filter tool."""
    parser.add_argument("--calls_vcf", help="VCF file with CNV calls", required=True, type=str)
    parser.add_argument("--output_vcf", help="Output VCF file", required=True, type=str)
    parser.add_argument("--ref_fasta", help="Reference genome FASTA file", required=True, type=str)


def __parse_args_annotate_regions(parser: argparse.ArgumentParser) -> None:
    """Add arguments specific to the region annotation tool."""
    parser.add_argument("--input_vcf", help="Input VCF file with CNV calls", required=True, type=str)
    parser.add_argument("--output_vcf", help="Output VCF file with region annotations", required=True, type=str)
    parser.add_argument(
        "--annotation_bed",
        help="BED file with region annotations (chr, start, end, annotation)",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--overlap_fraction",
        help="Minimum fraction of CNV that must overlap with annotation region (0.0-1.0)",
        required=False,
        type=float,
        default=0.5,
    )


def __parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse command-line arguments using subparsers for different tools.

    This allows each tool to have its own set of arguments while sharing
    common options like verbosity.

    To add a new tool:
    1. Create a new __parse_args_<toolname>() function with tool-specific arguments
    2. Add a new subparser below for the tool
    3. Add a new elif block in the run() function to handle the tool
    """
    # Main parser
    parser = argparse.ArgumentParser(
        prog="combine_cnmops_cnvpytor_cnv_calls.py",
        description="CNV processing toolkit - runs various CNV-related tools.",
    )

    # Common arguments across all tools
    parser.add_argument("--verbosity", help="Verbosity: ERROR, WARNING, INFO, DEBUG", required=False, default="INFO")

    # Create subparsers for different tools
    subparsers = parser.add_subparsers(dest="tool", help="Tool to run", required=True)

    # Concat tool subparser
    concat_parser = subparsers.add_parser(
        "concat",
        help="Combine CNV VCFs from different callers (cn.mops and cnvpytor)",
        description="Combines CNV VCF files from cn.mops and cnvpytor into a single sorted and indexed VCF.",
    )
    __parse_args_concat(concat_parser)

    dup_filter_parser = subparsers.add_parser(
        "filter_cnmops_dups",
        help="Filter short duplications from cn.mops calls in the combined CNV VCF",
        description="Adds CNMOPS_SHORT_DUPLICATION filter to short duplications in cn.mops calls.",
    )
    __parse_args_dup_length_filter(dup_filter_parser)

    gaps_perc_parser = subparsers.add_parser(
        "annotate_gaps",
        help="Annotate CNV calls with percentage of gaps (Ns) from reference genome",
        description="Annotates CNV calls in a VCF with the percentage of gaps (Ns) in the reference genome.",
    )
    __parse_args_gaps_perc(gaps_perc_parser)

    annotate_regions_parser = subparsers.add_parser(
        "annotate_regions",
        help="Annotate CNV calls with region annotations from BED file",
        description="Annotates CNV calls in a VCF with region-based annotations"
        " (e.g., telomere, centromere, low coverage).",
    )
    __parse_args_annotate_regions(annotate_regions_parser)

    return parser.parse_args(argv[1:])


def annotate_vcf_with_gap_perc(input_vcf: str, ref_fasta: str, output_vcf: str) -> None:
    """
    Annotate CNV VCF records with GAP_PERC INFO field representing the fraction of 'N' bases in the CNV region.

    Parameters
    ----------
    input_vcf : str
        Path to input VCF file containing CNV calls.
    ref_fasta : str
        Path to reference genome FASTA file. Should have .fai index.
    output_vcf : str
        Path to output VCF file with GAP_PERC annotation.
    """

    genome = Fasta(ref_fasta, rebuild=False, build_index=False)

    with pysam.VariantFile(input_vcf) as vcf_in:
        header = vcf_in.header
        if "GAP_PERC" not in header.info:
            header.info.add(*INFO_TAG_REGISTRY["GAP_PERC"][:-1])
        with pysam.VariantFile(output_vcf, "w", header=header) as vcf_out:
            for record in vcf_in:
                chrom = record.chrom
                start = record.start
                end = record.stop
                try:
                    seq_obj = genome[chrom][start : end + 1]
                    seq = seq_obj.seq if seq_obj is not None else ""
                    n_count = seq.upper().count("N")
                    region_len = end - start + 1
                    gap_perc = n_count / region_len if region_len > 0 else 0.0
                except Exception as e:
                    logger.warning(f"Could not retrieve sequence for {chrom}:{start}-{end}: {e}")
                    gap_perc = 0.0
                record.info["GAP_PERC"] = round(gap_perc, 5)
                vcf_out.write(record)
    VcfUtils().index_vcf(output_vcf)


def annotate_vcf_with_regions(
    input_vcf: str, annotation_bed: str, output_vcf: str, overlap_fraction: float = 0.5
) -> None:
    """
    Annotate CNV VCF records with region annotations from a BED file.

    For each CNV record, identifies overlapping annotation regions and collects their
    annotations. If the overlap fraction exceeds the threshold, all unique annotations
    are added to the REGION_ANNOTATIONS INFO field.

    Parameters
    ----------
    input_vcf : str
        Path to input VCF file containing CNV calls. Each record must have start and stop fields.
    annotation_bed : str
        Path to BED file with region annotations in format: chr, start, end, annotation.
        Annotations can contain multiple values separated by '|' (e.g., "Telomere_Centromere|Coverage-Mappability").
    output_vcf : str
        Path to output VCF file with REGION_ANNOTATIONS added to INFO field.
    overlap_fraction : float, optional
        Minimum fraction of CNV length that must overlap with annotation regions to
        collect annotations. Must be between 0.0 and 1.0. Default is 0.5.

    Notes
    -----
    - For each CNV record, calculates overlap with all annotation regions in the BED file
    - Collects annotations from regions where overlap exceeds overlap_fraction * CNV_length
    - All annotation values (split by '|') are collected and made unique
    - Final unique annotations are joined with '|' and added to REGION_ANNOTATIONS INFO field
    - If no annotations meet the threshold, REGION_ANNOTATIONS will be empty or not added

    Examples
    --------
    >>> annotate_vcf_with_regions(
    ...     input_vcf="cnv_calls.vcf.gz",
    ...     annotation_bed="genome_regions.bed",
    ...     output_vcf="cnv_calls.annotated.vcf.gz",
    ...     overlap_fraction=0.5
    ... )
    """
    # Create temporary directory in the same directory as output_vcf
    output_dir = os.path.dirname(output_vcf) or "."
    with tempfile.TemporaryDirectory(dir=output_dir) as tmpdir:
        # Step 1: Convert VCF to BED format with unique index-based identifiers
        logger.info("Converting VCF to BED format")
        vcf_bed = pjoin(tmpdir, "input_cnvs.bed")
        cnv_vcf_to_bed(input_vcf, vcf_bed, assign_id=True)

        # Step 2: Run bedtools coverage to calculate total overlap statistics
        # bedtools coverage output adds these columns to each interval in A:
        # - Number of features in B that overlapped
        # - Number of bases in A that had coverage from B
        # - Length of A interval
        # - Fraction of A that had coverage from B (this is what we need!)
        logger.info("Running bedtools coverage to calculate total overlaps")
        coverage_bed = pjoin(tmpdir, "coverage.bed")
        bed_utils = BedUtils()
        bed_utils.bedtools_coverage(vcf_bed, annotation_bed, coverage_bed)

        # Step 3: Parse coverage results to identify CNVs meeting the overlap threshold
        logger.info("Identifying CNVs meeting overlap threshold")
        cnv_indices_meeting_threshold = set()
        with open(coverage_bed) as cov_fh:
            for line in cov_fh:
                fields = line.rstrip().split("\t")
                record_id = fields[3]  # The unique identifier (e.g., CNV_000000001)
                idx = int(record_id.split("_")[1])  # Extract numeric index
                overlap_fraction_actual = float(fields[7])  # Fraction of bases in A with coverage from all B features

                if overlap_fraction_actual >= overlap_fraction:
                    cnv_indices_meeting_threshold.add(idx)

        # Step 4: Collect annotations for CNVs meeting threshold using bedtools map
        # bedtools map will aggregate all annotation values from overlapping B features
        logger.info("Collecting annotations for CNVs meeting threshold")
        map_bed = pjoin(tmpdir, "map.bed")
        # Use collapse operation to collect all annotation values (4th column) separated by '|'
        bed_utils.bedtools_map(
            vcf_bed, annotation_bed, map_bed, column=4, operation="collapse", additional_args='-delim "|"'
        )

        # Parse map results to collect annotations only for CNVs meeting threshold
        cnv_annotations = {}  # Maps record_index -> set of annotation values
        with open(map_bed) as map_fh:
            for line in map_fh:
                fields = line.rstrip().split("\t")
                record_id = fields[3]  # From A (CNV) - format: CNV_000000001
                idx = int(record_id.split("_")[1])  # Extract the numeric index

                # Only collect annotations for CNVs that met the threshold
                if idx not in cnv_indices_meeting_threshold:
                    continue

                # The mapped column contains '|'-separated annotation values
                mapped_values = fields[4]
                if mapped_values != ".":  # bedtools map uses "." for no overlap
                    cnv_annotations[idx] = set(mapped_values.split("|"))

        # Step 5: Write output VCF with REGION_ANNOTATIONS
        logger.info("Writing annotated VCF")
        with pysam.VariantFile(input_vcf) as vcf_in:
            header = vcf_in.header

            # Add REGION_ANNOTATIONS INFO field if not present
            if "REGION_ANNOTATIONS" not in header.info:
                header.info.add(*INFO_TAG_REGISTRY["REGION_ANNOTATIONS"][:-1])

            with pysam.VariantFile(output_vcf, "w", header=header) as vcf_out:
                for record_index, record in enumerate(vcf_in):
                    # Add annotations if this CNV has any
                    if record_index in cnv_annotations and cnv_annotations[record_index]:
                        # Get existing annotations if present
                        existing_annotations = set()
                        if "REGION_ANNOTATIONS" in record.info:
                            existing_annotations = set(record.info["REGION_ANNOTATIONS"])

                        # Combine existing and new annotations
                        all_annotations = sorted(existing_annotations | cnv_annotations[record_index])
                        record.info["REGION_ANNOTATIONS"] = all_annotations

                    vcf_out.write(record)

    VcfUtils().index_vcf(output_vcf)
    logger.info(f"Successfully annotated VCF with regions: {output_vcf}")


def combine_cnv_vcfs(
    cnmops_vcf: str,
    cnvpytor_vcf: str,
    fasta_index: str,
    output_vcf: str,
    output_directory: str | None = None,
) -> str:
    """
    Combine VCF files from cn.mops and CNVpytor into a single sorted and indexed VCF.

    This function performs the following steps:
    1. Updates headers of both VCFs to contain the same contigs from the FASTA index
    2. Combines the headers from both updated files (excluding FILTER fields)
    3. Adds an INFO tag for the source (CNV_SOURCE) to identify the caller
    4. Writes records from both VCF files to the combined output
    5. Sorts and indexes the final VCF

    Note: This function does NOT merge overlapping CNV records - it simply concatenates
    all records from both input files.

    Parameters
    ----------
    cnmops_vcf : str
        Path to the cn.mops VCF file (.vcf.gz)
    cnvpytor_vcf : str
        Path to the CNVpytor VCF file (.vcf.gz)
    fasta_index : str
        Path to the reference genome FASTA index file (.fai)
    output_vcf : str
        Path to the output combined VCF file (.vcf.gz)
    output_directory : str, optional
        Directory for storing temporary files

    Returns
    -------
    str
        Path to the final sorted and indexed combined VCF file

    Raises
    ------
    FileNotFoundError
        If any of the input files do not exist
    RuntimeError
        If VCF processing fails

    Examples
    --------
    >>> combined_vcf = combine_cnv_vcfs(
    ...     cnmops_vcf="cnmops.vcf.gz",
    ...     cnvpytor_vcf="cnvpytor.vcf.gz",
    ...     fasta_index="genome.fa.fai",
    ...     output_vcf="combined.vcf.gz",
    ...     output_directory="/tmp/cnv_combine"
    ... )
    """
    # Validate input files exist
    for input_file in [cnmops_vcf, cnvpytor_vcf, fasta_index]:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file does not exist: {input_file}")

    # Create output directory if it doesn't exist
    if output_directory is None:
        output_directory = os.path.dirname(output_vcf)
    if output_directory:  # file with no directory evaluates to ""
        os.makedirs(output_directory, exist_ok=True)

    vcf_utils = VcfUtils()

    # Step 1: Update headers to contain same contigs from FASTA index
    cnmops_vcf_updated, cnvpytor_vcf_updated = update_vcf_contigs(
        vcf_utils, cnmops_vcf, cnvpytor_vcf, fasta_index, output_directory
    )

    # Step 2: Open updated VCF files, combine headers (excluding FILTER fields), and add CNV_SOURCE tag
    logger.info("Combining VCF headers and adding CNV_SOURCE INFO tag")
    with pysam.VariantFile(cnmops_vcf_updated) as vcf1, pysam.VariantFile(cnvpytor_vcf_updated) as vcf2:
        # Combine headers (excluding FILTER fields)
        combined_header = combine_vcf_headers_for_cnv(vcf1.header, vcf2.header)

        # Add INFO tag for source if not already present
        if "CNV_SOURCE" not in combined_header.info:
            combined_header.info.add(*INFO_TAG_REGISTRY["CNV_SOURCE"][:-1])

        # Step 3: Write records from both VCF files
        logger.info("Writing records from both VCF files to temporary combined VCF")
        temp_combined_vcf = pjoin(output_directory, "temp_combined.vcf.gz")

        with pysam.VariantFile(temp_combined_vcf, "w", header=combined_header) as vcf_out:
            write_vcf_records_with_source(vcf1, vcf_out, combined_header, "cn.mops")
            write_vcf_records_with_source(vcf2, vcf_out, combined_header, "cnvpytor")

    # Step 4: Sort and index the VCF
    logger.info("Sorting and indexing the combined VCF")
    vcf_utils.sort_vcf(temp_combined_vcf, output_vcf)
    vcf_utils.index_vcf(output_vcf)

    # Clean up temporary files
    mu.cleanup_temp_files([cnmops_vcf_updated, cnvpytor_vcf_updated, temp_combined_vcf])

    logger.info(f"Successfully created combined VCF: {output_vcf}")
    return output_vcf


def filter_dup_cnmmops_cnv_calls(
    combined_calls: str, combined_calls_annotated: str, filtered_length: str, distance_threshold: int
) -> None:
    """
    Collapses adjacent cnmops duplications with distance less than distance_threshold
    Adds CNMOPS_SHORT_DUPLICATION filter to the short duplications (less than filtered_length) that cn.mops returns.

    Parameters
    ----------
    combined_calls : str
        Path to the combined cn.mops and cnvpytor CNV calls bed file.
    combined_calls_annotated : str
        Path to the combined cn.mops and cnvpytor CNV calls bed file with annotations.
    filtered_length : str
        Minimum duplication length to be considered valid.
    distance_threshold : int
        Distance threshold for merging CNV segments.
    """
    output_dir = os.path.dirname(combined_calls_annotated)
    temporary_files = []
    vu = VcfUtils()

    deletion_vcf = pjoin(output_dir, "temp_deletions.vcf.gz")
    vu.view_vcf(combined_calls, deletion_vcf, extra_args="-e \"(INFO/SVTYPE='DUP') && (INFO/CNV_SOURCE='cn.mops')\"")
    vu.index_vcf(deletion_vcf)
    temporary_files.append(deletion_vcf)

    duplication_vcf = pjoin(output_dir, "temp_duplications.vcf.gz")
    vu.view_vcf(combined_calls, duplication_vcf, extra_args="-i \"(INFO/SVTYPE='DUP') && (INFO/CNV_SOURCE='cn.mops')\"")
    vu.index_vcf(duplication_vcf)
    temporary_files.append(duplication_vcf)

    collapsed_duplication_vcf = pjoin(output_dir, "temp_collapsed_duplications.vcf.gz")
    merge_cnvs_in_vcf(duplication_vcf, collapsed_duplication_vcf, distance=distance_threshold)
    vu.index_vcf(collapsed_duplication_vcf)
    temporary_files.append(collapsed_duplication_vcf)

    combined_calls = pjoin(output_dir, "temp_combined_calls.vcf.gz")
    vu.concat_vcf([deletion_vcf, collapsed_duplication_vcf], combined_calls)
    vu.index_vcf(combined_calls)
    temporary_files.append(combined_calls)

    with pysam.VariantFile(combined_calls) as vcf_in:
        hdr = vcf_in.header
        hdr.filters.add(*FILTER_TAG_REGISTRY["CNMOPS_SHORT_DUPLICATION"][:-1])
        with pysam.VariantFile(combined_calls_annotated, "w", header=hdr) as vcf_out:
            for record in vcf_in:
                if record.info["SVTYPE"] == "DUP":
                    svlen = abs(record.info.get("SVLEN", [0])[0])
                    if svlen < int(filtered_length):
                        if "FILTER" in record.filter.keys():
                            record.filter.add("CNMOPS_SHORT_DUPLICATION")
                        else:
                            record.filter.add("CNMOPS_SHORT_DUPLICATION")
                vcf_out.write(record)
    vu.index_vcf(combined_calls_annotated)
    mu.cleanup_temp_files(temporary_files)


def process_del_jalign_results(
    del_jalign_results: str,
    sample_name: str,
    out_directory: str,
    ref_fasta: str,
    pN: float = 0,  # noqa: N803
) -> str:
    """
    Processes jalign results for deletions and filters them.

    Parameters
    ----------
    del_jalign_results : str
        Jalign results for Deletions in tsv format.
    sample_name : str
        Sample name.
    out_directory : str
        Output folder to store results.
    ref_fasta : str
        Reference genome fasta file.
    pN : float, optional
        Threshold for filtering CNV calls based on the fraction of reference genome gaps (Ns) in the call region.

    Returns
    -------
    str
        Path to deletions called by cn.mops and cnvpytor bed file.
    """
    # reads jalign results
    df_cnmops_cnvpytor_del = pd.read_csv(del_jalign_results, sep="\t", header=None)
    df_cnmops_cnvpytor_del.columns = [
        "chrom",
        "start",
        "end",
        "CN",
        "jalign_written",
        "6",
        "7",
        "jdelsize_min",
        "jdelsize_max",
        "jdelsize_avg",
        "jumpland_min",
        "jumpland_max",
        "jumpland_avg",
    ]
    df_cnmops_cnvpytor_del["len"] = df_cnmops_cnvpytor_del["end"] - df_cnmops_cnvpytor_del["start"]
    df_cnmops_cnvpytor_del["source"] = df_cnmops_cnvpytor_del["CN"].apply(
        lambda x: "cn.mops" if pd.Series(x).str.contains("CN").any() else "cnvpytor"
    )
    df_cnmops_cnvpytor_del["CNV_type"] = "DEL"
    df_cnmops_cnvpytor_del["copy_number"] = df_cnmops_cnvpytor_del["CN"].apply(lambda x: x.replace("CN", ""))
    df_cnmops_cnvpytor_del["copy_number"] = df_cnmops_cnvpytor_del["copy_number"].apply(
        lambda x: "DEL" if pd.Series(x).str.contains("deletion").any() else x
    )

    df_cnmops_cnvpytor_del_filtered = df_cnmops_cnvpytor_del[df_cnmops_cnvpytor_del["pN"] <= pN]

    out_del_jalign = pjoin(
        out_directory,
        f"{sample_name}.cnmops_cnvpytor.DEL.jalign.bed",
    )
    df_cnmops_cnvpytor_del_filtered[
        ["chrom", "start", "end", "CNV_type", "source", "copy_number", "jalign_written"]
    ].to_csv(out_del_jalign, sep="\t", header=False, index=False)

    out_del_jalign_merged = pjoin(
        out_directory,
        f"{sample_name}.cnmops_cnvpytor.DEL.jalign.merged.bed",
    )

    run_cmd(
        f"cat {out_del_jalign} | bedtools sort -i - | \
            bedtools merge -c 4,5,6,7 -o distinct,distinct,distinct,max  -i -  > {out_del_jalign_merged}"
    )

    return out_del_jalign_merged


def run(argv: list[str]):
    """
    Driver function for CNV processing tools.

    This function routes execution to the appropriate tool based on the first argument.
    Currently supported tools:
    - concat: Combines CNV VCFs from cn.mops and cnvpytor into a single sorted VCF

    The function can be called:
    1. As a standalone script: python combine_cnmops_cnvpytor_cnv_calls.py concat --cnmops_vcf ... --cnvpytor_vcf ...
    2. As part of combine_cnmops_cnvpytor_cnv_calls: combine_cnmops_cnvpytor_cnv_calls concat --cnmops_vcf ...

    Parameters
    ----------
    argv : list of str
        Command-line arguments where argv[1] is the tool name (e.g., 'concat')
        and remaining arguments are tool-specific parameters.

    Examples
    --------
    >>> run(['prog', 'concat', '--cnmops_vcf', 'cnmops.vcf.gz',
    ...      '--cnvpytor_vcf', 'cnvpytor.vcf.gz', '--output_vcf', 'combined.vcf.gz',
    ...      '--fasta_index', 'genome.fa.fai', '--out_directory', '/tmp'])
    """
    args = __parse_args(argv)
    logger.setLevel(getattr(logging, args.verbosity))

    if args.tool == "concat":
        combine_cnv_vcfs(
            cnmops_vcf=args.cnmops_vcf,
            cnvpytor_vcf=args.cnvpytor_vcf,
            fasta_index=args.fasta_index,
            output_vcf=args.output_vcf,
            output_directory=args.out_directory,
        )
    elif args.tool == "filter_cnmops_dups":
        filter_dup_cnmmops_cnv_calls(
            combined_calls=args.combined_calls,
            combined_calls_annotated=args.combined_calls_annotated,
            filtered_length=args.filtered_length,
            distance_threshold=args.distance_threshold,
        )
    elif args.tool == "annotate_gaps":
        annotate_vcf_with_gap_perc(
            input_vcf=args.calls_vcf,
            ref_fasta=args.ref_fasta,
            output_vcf=args.output_vcf,
        )
    elif args.tool == "annotate_regions":
        annotate_vcf_with_regions(
            input_vcf=args.input_vcf,
            annotation_bed=args.annotation_bed,
            output_vcf=args.output_vcf,
            overlap_fraction=args.overlap_fraction,
        )
    else:
        raise ValueError(f"Unknown tool: {args.tool}")


def main():
    run(sys.argv)


def main_concat():
    """
    Entry point for standalone combine_cnv_vcfs script.

    This allows running the concat tool directly without specifying the tool name:
    combine_cnv_vcfs --cnmops_vcf ... --cnvpytor_vcf ...

    Instead of:
    combine_cnmops_cnvpytor_cnv_calls concat --cnmops_vcf ... --cnvpytor_vcf ...
    """
    # Insert 'concat' as the tool argument
    argv = [sys.argv[0], "concat"] + sys.argv[1:]
    run(argv)


def main_filter_dup_cnmmops():
    """
    Entry point for standalone filter_dup_cnmmops_cnv_calls script.

    This allows running the filter_dup_cnmmops tool directly without specifying the tool name:
    filter_dup_cnmmops_cnv_calls --combined_calls ... --combined_calls_annotated ...

    Instead of:
    combine_cnmops_cnvpytor_cnv_calls filter_dup_cnmmops --combined_calls ... --combined_calls_annotated ...
    """
    # Insert 'filter_dup_cnmmops' as the tool argument
    argv = [sys.argv[0], "filter_cnmops_dups"] + sys.argv[1:]
    run(argv)


def main_gaps_perc():
    """
    Entry point for standalone annotate_vcf_with_gap_perc script.

    This allows running the annotate_gaps tool directly without specifying the tool name:
    annotate_gaps --input_vcf ... --ref_fasta ... --output_vcf ...

    Instead of:
    combine_cnmops_cnvpytor_cnv_calls annotate_gaps --input_vcf ... --ref_fasta ... --output_vcf ...
    """
    # Insert 'annotate_gaps' as the tool argument
    argv = [sys.argv[0], "annotate_gaps"] + sys.argv[1:]
    run(argv)


def main_annotate_regions():
    """
    Entry point for standalone annotate_vcf_with_regions script.

    This allows running the annotate_regions tool directly without specifying the tool name:
    annotate_regions --input_vcf ... --annotation_bed ... --output_vcf ... --overlap_fraction ...

    Instead of:
    combine_cnmops_cnvpytor_cnv_calls annotate_regions --input_vcf ... --annotation_bed ... --output_vcf ...
    """
    # Insert 'annotate_regions' as the tool argument
    argv = [sys.argv[0], "annotate_regions"] + sys.argv[1:]
    run(argv)


if __name__ == "__main__":
    main()
