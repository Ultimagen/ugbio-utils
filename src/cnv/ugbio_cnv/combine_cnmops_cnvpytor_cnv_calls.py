import argparse
import logging
import os
import subprocess
import sys
import tempfile
from os.path import join as pjoin

import pysam
from pyfaidx import Fasta
from ugbio_cnv.analyze_cnv_breakpoint_reads import analyze_cnv_breakpoints
from ugbio_cnv.analyze_cnv_breakpoint_reads import get_parser as get_breakpoint_parser
from ugbio_cnv.cnv_vcf_consts import INFO_TAG_REGISTRY
from ugbio_cnv.combine_cnv_vcf_utils import (
    cnv_vcf_to_bed,
    combine_cnv_or_sv_vcf_files,
    merge_cnvs_in_vcf,
)
from ugbio_cnv.merge_cnv_sv import merge_cnv_sv_vcfs
from ugbio_core.bed_utils import BedUtils
from ugbio_core.logger import logger
from ugbio_core.vcf_utils import VcfUtils


def run_cmd(cmd):
    logger.info(cmd)
    subprocess.run(cmd, shell=True, check=True)  # noqa: S602


def __parse_args_concat(parser: argparse.ArgumentParser) -> None:
    """Add arguments specific to the concat tool."""
    parser.add_argument(
        "--cnmops_vcf",
        help="input VCF file(s) holding cn.mops CNV calls",
        required=False,
        type=str,
        nargs="*",
        default=[],
    )
    parser.add_argument(
        "--cnvpytor_vcf",
        help="input VCF file(s) holding cnvpytor CNV calls",
        required=False,
        type=str,
        nargs="*",
        default=[],
    )
    parser.add_argument("--output_vcf", help="output combined VCF file", required=True, type=str)
    parser.add_argument("--fasta_index", help="fasta.fai file", required=True, type=str)
    parser.add_argument("--out_directory", help="output directory", required=False, type=str)
    parser.add_argument(
        "--make_ids_unique",
        help="ensure all variant IDs are unique by appending suffixes (e.g., ID_1, ID_2) when duplicates exist",
        action="store_true",
        default=False,
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
    parser.add_argument(
        "--genome",
        help="Genome file (chr tab length or .fai)",
        required=True,
        type=str,
    )


def __parse_args_merge_records(parser: argparse.ArgumentParser) -> None:
    """Add arguments specific to the merge records tool."""
    parser.add_argument("--input_vcf", help="Input VCF file with CNV calls to merge", required=True, type=str)
    parser.add_argument("--output_vcf", help="Output VCF file with merged CNV calls", required=True, type=str)
    parser.add_argument(
        "--distance",
        help="Distance threshold for merging CNV segments (default: 0)",
        required=False,
        type=int,
        default=0,
    )
    parser.add_argument(
        "--enable_smoothing",
        help="Enable size-scaled gap CNV smoothing (BIOIN-2622). When enabled, large CNVs can be merged "
        "across larger gaps while preventing inappropriate merging of small CNVs. Requires CIPOS INFO field.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--max_gap_absolute",
        help="Absolute maximum gap for merging CNVs in smoothing mode (bp). "
        "This caps the maximum gap regardless of CNV size. Default: 50000 (50kb)",
        type=int,
        default=50000,
    )
    parser.add_argument(
        "--gap_scale_fraction",
        help="Gap as fraction of smaller CNV length for smoothing. "
        "Formula: gap_threshold = min(max_gap_absolute, gap_scale_fraction × larger_CNV). "
        "Default: 0.05 (5%%)",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--cipos_threshold",
        help="Minimum CIPOS length for smoothing (bp). CNVs with CIPOS length < threshold have "
        "high-confidence breakpoints and won't be smoothed. CIPOS length = max - min - 1. Default: 50",
        type=int,
        default=50,
    )


def __parse_args_merge_cnv_sv(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the merge_cnv_sv tool."""
    parser.add_argument("--cnv_vcf", help="Input CNV VCF file", required=True, type=str)
    parser.add_argument("--sv_vcf", help="Input SV VCF file (e.g., from GRIDSS)", required=True, type=str)
    parser.add_argument("--output_vcf", help="Output merged VCF file", required=True, type=str)
    parser.add_argument("--fasta_index", help="Reference genome FASTA index (.fai) file", required=True, type=str)
    parser.add_argument(
        "--min_sv_length",
        help="Minimum absolute SVLEN for SV calls to include (default: 1000)",
        required=False,
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--max_sv_length",
        help="Maximum absolute SVLEN for SV calls to include (default: 5000000, 5Mb)",
        required=False,
        type=int,
        default=5000000,
    )
    parser.add_argument(
        "--min_sv_qual",
        help="Minimum QUAL score for SV calls to include (default: 0, no minimum)",
        required=False,
        type=float,
        default=0,
    )
    parser.add_argument(
        "--distance",
        help="Distance threshold for collapsing overlapping variants (default: 0, exact overlaps only)",
        required=False,
        type=int,
        default=0,
    )
    parser.add_argument(
        "--pctsize",
        help="Minimum size similarity (0.0-1.0) for collapsing (default: 0.5, require 50%% match)",
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

    merge_records_parser = subparsers.add_parser(
        "merge_records",
        help="Merge adjacent or nearby CNV records in a VCF file",
        description="Merges CNV records that are within a specified distance threshold.",
    )
    __parse_args_merge_records(merge_records_parser)

    analyze_breakpoints_parser = subparsers.add_parser(
        "analyze_breakpoint_reads",
        help="Analyze reads at CNV breakpoints for DUP/DEL evidence",
        description="Annotates CNV VCF with breakpoint read support information.",
    )
    # Reuse argument definitions from analyze_cnv_breakpoint_reads module
    get_breakpoint_parser(analyze_breakpoints_parser)

    merge_cnv_sv_parser = subparsers.add_parser(
        "merge_cnv_sv",
        help="Merge CNV VCF with filtered SV VCF, replacing overlapping CNVs with SV calls",
        description="Filters SV VCF for PASS DEL/DUP calls above length threshold, "
        "then merges with CNV VCF, replacing overlapping CNV calls with higher-quality SV calls.",
    )
    __parse_args_merge_cnv_sv(merge_cnv_sv_parser)

    return parser.parse_args(argv[1:])


def annotate_vcf_with_gap_perc(input_vcf: str, ref_fasta: str, output_vcf: str) -> None:
    """
    Annotate CNV VCF records with GAP_PERCENTAGE INFO field representing the fraction of 'N' bases in the CNV region.

    Parameters
    ----------
    input_vcf : str
        Path to input VCF file containing CNV calls.
    ref_fasta : str
        Path to reference genome FASTA file. Should have .fai index.
    output_vcf : str
        Path to output VCF file with GAP_PERCENTAGE annotation.
    """

    genome = Fasta(ref_fasta, rebuild=False, build_index=False)

    with pysam.VariantFile(input_vcf) as vcf_in:
        header = vcf_in.header
        if "GAP_PERCENTAGE" not in header.info:
            header.info.add(*INFO_TAG_REGISTRY["GAP_PERCENTAGE"][:-1])
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
                record.info["GAP_PERCENTAGE"] = round(gap_perc, 5)
                vcf_out.write(record)
    VcfUtils().index_vcf(output_vcf)


def annotate_vcf_with_regions(
    input_vcf: str, annotation_bed: str, output_vcf: str, genome: str, overlap_fraction: float = 0.5
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
    genome: str
        Genome file (chr tab length or .fai)

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
    ...     geome = "Homo_sapiens_assembly38.fasta",
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
            vcf_bed, annotation_bed, map_bed, column=4, operation="collapse", additional_args=f'-delim "|" -g {genome}'
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
    cnmops_vcf: list[str],
    cnvpytor_vcf: list[str],
    fasta_index: str,
    output_vcf: str,
    output_directory: str | None = None,
    *,
    make_ids_unique: bool = False,
) -> str:
    """
    Concatenates VCF files from cn.mops and CNVpytor into a single sorted and indexed VCF.

    This function performs the following steps:
    1. Updates headers of all VCFs to contain the same contigs from the FASTA index
    2. Combines the headers from all updated files (excluding FILTER fields)
    3. Adds an INFO tag for the source (CNV_SOURCE) to identify the caller
    4. Writes records from all VCF files to the combined output
    5. Sorts and indexes the final VCF

    Note: This function does NOT merge overlapping CNV records - it simply concatenates
    all records from all input files.

    Parameters
    ----------
    cnmops_vcf : list[str]
        List of paths to cn.mops VCF files (.vcf.gz). Can be empty.
    cnvpytor_vcf : list[str]
        List of paths to CNVpytor VCF files (.vcf.gz). Can be empty.
    fasta_index : str
        Path to the reference genome FASTA index file (.fai)
    output_vcf : str
        Path to the output combined VCF file (.vcf.gz)
    output_directory : str, optional
        Directory for storing temporary files
    make_ids_unique : bool, optional
        If True, assign unique IDs to all variants (default: False)

    Returns
    -------
    str
        Path to the final sorted and indexed combined VCF file

    Raises
    ------
    FileNotFoundError
        If any of the input files do not exist
    ValueError
        If both cnmops_vcf and cnvpytor_vcf are empty
    RuntimeError
        If VCF processing fails

    Examples
    --------
    >>> combined_vcf = combine_cnv_vcfs(
    ...     cnmops_vcf=["cnmops.vcf.gz"],
    ...     cnvpytor_vcf=["cnvpytor1.vcf.gz", "cnvpytor2.vcf.gz"],
    ...     fasta_index="genome.fa.fai",
    ...     output_vcf="combined.vcf.gz",
    ...     output_directory="/tmp/cnv_combine",
    ...     make_ids_unique=True
    ... )

    See Also
    --------
    combine_cnv_or_sv_vcf_files : More flexible VCF combination accepting list of tuples
    """
    # Validate that at least one VCF list is not empty
    if not cnmops_vcf and not cnvpytor_vcf:
        raise ValueError("At least one of cnmops_vcf or cnvpytor_vcf must be non-empty")

    # Build list of (vcf_path, source_name) tuples
    vcf_files = [(vcf, "cn.mops") for vcf in cnmops_vcf] + [(vcf, "cnvpytor") for vcf in cnvpytor_vcf]

    # Call unified function
    return combine_cnv_or_sv_vcf_files(
        vcf_files=vcf_files,
        output_vcf=output_vcf,
        fasta_index=fasta_index,
        preserve_filters=False,  # Clear filters to PASS (from what the sub-pipelines may be setting)
        make_ids_unique=make_ids_unique,
        output_directory=output_directory,
    )


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
    ...      '--cnvpytor_vcf', 'cnvpytor1.vcf.gz', 'cnvpytor2.vcf.gz',
    ...      '--output_vcf', 'combined.vcf.gz',
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
            make_ids_unique=args.make_ids_unique,
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
            genome=args.genome,
        )
    elif args.tool == "merge_records":
        merge_cnvs_in_vcf(
            input_vcf=args.input_vcf,
            output_vcf=args.output_vcf,
            distance=args.distance,
            ignore_filter=False,
            ignore_sv_type=True,
            pick_best=True,
            enable_smoothing=args.enable_smoothing,
            max_gap_absolute=args.max_gap_absolute,
            gap_scale_fraction=args.gap_scale_fraction,
            cipos_threshold=args.cipos_threshold,
        )
    elif args.tool == "analyze_breakpoint_reads":
        analyze_cnv_breakpoints(
            bam_file=args.bam_file,
            vcf_file=args.vcf_file,
            cushion=args.cushion,
            output_file=args.output_file,
            reference_fasta=args.reference_fasta,
            output_bam=args.output_bam,
        )
    elif args.tool == "merge_cnv_sv":
        merge_cnv_sv_vcfs(
            cnv_vcf=args.cnv_vcf,
            sv_vcf=args.sv_vcf,
            output_vcf=args.output_vcf,
            fasta_index=args.fasta_index,
            min_sv_length=args.min_sv_length,
            max_sv_length=args.max_sv_length,
            min_sv_qual=args.min_sv_qual,
            distance=args.distance,
            pctsize=args.pctsize,
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


def main_merge_records():
    """
    Entry point for standalone merge_cnvs_in_vcf script.

    This allows running the merge_records tool directly without specifying the tool name:
    merge_records --input_vcf ... --output_vcf ... --distance ...

    Instead of:
    combine_cnmops_cnvpytor_cnv_calls merge_records --input_vcf ... --output_vcf ... --distance ...
    """
    # Insert 'merge_records' as the tool argument
    argv = [sys.argv[0], "merge_records"] + sys.argv[1:]
    run(argv)


def main_analyze_breakpoints():
    """
    Entry point for standalone analyze_breakpoint_reads script.

    This allows running the analyze_breakpoint_reads tool directly without specifying the tool name:
    analyze_cnv_breakpoint_reads --bam-file ... --vcf-file ... --output-file ...

    Instead of:
    combine_cnmops_cnvpytor_cnv_calls analyze_breakpoint_reads --bam-file ... --vcf-file ...
    """
    # Insert 'analyze_breakpoint_reads' as the tool argument
    argv = [sys.argv[0], "analyze_breakpoint_reads"] + sys.argv[1:]
    run(argv)


def main_merge_cnv_sv():
    """
    Entry point for standalone merge_cnv_sv script.

    This allows running merge_cnv_sv directly:
    merge_cnv_sv --cnv_vcf ... --sv_vcf ... --output_vcf ...

    Instead of:
    combine_cnmops_cnvpytor_cnv_calls merge_cnv_sv --cnv_vcf ... --sv_vcf ...
    """
    # Insert 'merge_cnv_sv' as the tool argument
    argv = [sys.argv[0], "merge_cnv_sv"] + sys.argv[1:]
    run(argv)


if __name__ == "__main__":
    main()
