import argparse
import logging
import os
import subprocess
import sys
from os.path import join as pjoin

import pandas as pd
import pysam
import ugbio_core.misc_utils as mu
from pyfaidx import Fasta
from ugbio_core.logger import logger
from ugbio_core.vcf_utils import VcfUtils

bedmap = "bedmap"


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

    return parser.parse_args(argv[1:])


def annotate_vcf_with_gap_perc(input_vcf: str, ref_fasta: str, output_vcf: str) -> None:
    """
    Annotate CNV VCF records with GAP_PERC INFO field representing the fraction of 'N' bases in the CNV region.

    Parameters
    ----------
    input_vcf : str
        Path to input VCF file containing CNV calls.
    ref_fasta : str
        Path to reference genome FASTA file.
    output_vcf : str
        Path to output VCF file with GAP_PERC annotation.
    """

    genome = Fasta(ref_fasta)

    with pysam.VariantFile(input_vcf) as vcf_in:
        header = vcf_in.header
        if "GAP_PERC" not in header.info:
            header.info.add(
                "GAP_PERC",
                number=1,
                type="Float",
                description="Fraction of N bases in the CNV region from reference genome",
            )
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


# Helper function to check and add metadata records
def _add_metadata_records(
    record_dict1: pysam.VariantHeaderMetadata,
    record_dict2: pysam.VariantHeaderMetadata,
    record_type: str,
    enforced_info_specs: dict,
    combined_header: pysam.VariantHeader,
) -> None:
    """Add metadata records (INFO/FORMAT/FILTER) to combined header. (service function), modifies the header in place
    Parameters
    ----------
    record_dict1 : pysam.VariantHeaderMetadata
        Dictionary of metadata records from the first header
    record_dict2 : pysam.VariantHeaderMetadata
        Dictionary of metadata records from the second header
    record_type : str
        Type of metadata records ("INFO", "FORMAT", or "FILTER")
    enforced_info_specs : dict
        Dictionary of enforced specifications for certain INFO fields
    combined_header : pysam.VariantHeader
        The combined VCF header to add records to
    """

    # Collect all unique keys from both headers
    all_keys = set(record_dict1.keys()) | set(record_dict2.keys())

    for key in all_keys:
        # Check if this field has enforced specifications
        if record_type == "INFO" and key in enforced_info_specs:
            number, data_type, description = enforced_info_specs[key]
            combined_header.info.add(key, number, data_type, description)
            continue

        # Handle fields that exist in both headers
        if key in record_dict1 and key in record_dict2:
            record1 = record_dict1[key]
            record2 = record_dict2[key]

            # Check if type or number differ
            if hasattr(record1, "type") and hasattr(record2, "type"):
                if record1.type != record2.type:
                    msg = f"{record_type} field '{key}' has conflicting types: '{record1.type}' vs '{record2.type}'"
                    raise RuntimeError(msg)
            if hasattr(record1, "number") and hasattr(record2, "number"):
                if record1.number != record2.number:
                    msg = (
                        f"{record_type} field '{key}' has conflicting numbers: '{record1.number}' vs '{record2.number}'"
                    )
                    raise RuntimeError(msg)

            # If we get here, they're compatible - use the first one
            combined_header.add_line(str(record1.record))

        # Handle fields that only exist in header1
        elif key in record_dict1:
            combined_header.add_line(str(record_dict1[key].record))

        # Handle fields that only exist in header2
        elif key in record_dict2:
            combined_header.add_line(str(record_dict2[key].record))


def combine_vcf_headers_for_cnv(
    header1: pysam.VariantHeader, header2: pysam.VariantHeader, *, keep_filters: bool = False
) -> pysam.VariantHeader:
    """
    Combine two VCF headers into a single header for CNV/SV variant calls.

    This function is specifically designed for merging headers from different CNV/SV
    callers (e.g., cn.mops and CNVpytor). It creates a new header that includes all
    metadata from both input headers with special handling for structural variant fields.

    For INFO and FORMAT fields with the same ID:
    - If type and number match, use the first definition
    - If type or number differ, raise RuntimeError

    FILTER fields are cleared unless keep_filters is set to True.

    Special enforcement for CNV/SV-related fields to ensure VCF spec compliance:
    - SVLEN: enforces Number="." (variable-length array) per VCF 4.2 spec for structural variants
    - SVTYPE: enforces Number=1 (single value) per VCF 4.2 spec for structural variants

    Parameters
    ----------
    header1 : pysam.VariantHeader
        The first VCF header (takes precedence in case of identical collisions)
    header2 : pysam.VariantHeader
        The second VCF header to merge
    keep_filters : bool, optional
        Whether to keep FILTER fields from both headers. Default is False (FILTERs are cleared).

    Returns
    -------
    pysam.VariantHeader
        A new combined VCF header with unified metadata

    Raises
    ------
    RuntimeError
        If INFO/FORMAT IDs collide with different type or number specifications
        (except for SVLEN and SVTYPE which are enforced to standard values)

    Examples
    --------
    >>> import pysam
    >>> header1 = pysam.VariantFile("cnmops.vcf.gz").header
    >>> header2 = pysam.VariantFile("cnvpytor.vcf.gz").header
    >>> combined = combine_vcf_headers_for_cnv(header1, header2)
    """
    # Enforced specifications for CNV/SV fields per VCF 4.2 spec
    enforced_info_specs = {
        "SVLEN": (".", "Integer", "Length of structural variant (SV) per VCF 4.2 specification"),
        "SVTYPE": (1, "String", "Type of structural variant (SV); e.g. DEL, DUP, etc. (per VCF 4.2 specification)"),
    }

    # Start with a copy of the first header
    combined_header = pysam.VariantHeader()

    # Copy contigs from header1 (assuming they're the same)
    for contig in header1.contigs.values():
        combined_header.contigs.add(contig.name, length=contig.length)

    # Add INFO fields
    _add_metadata_records(header1.info, header2.info, "INFO", enforced_info_specs, combined_header)

    # Add FORMAT fields
    _add_metadata_records(header1.formats, header2.formats, "FORMAT", enforced_info_specs, combined_header)

    if keep_filters:
        # Add FILTER fields
        _add_metadata_records(header1.filters, header2.filters, "FILTER", {}, combined_header)

    # Add samples from both headers
    for sample in header1.samples:
        if sample not in combined_header.samples:
            combined_header.add_sample(sample)
    for sample in header2.samples:
        if sample not in combined_header.samples:
            combined_header.add_sample(sample)

    return combined_header


def _update_vcf_contigs(
    vcf_utils: VcfUtils,
    cnmops_vcf: str,
    cnvpytor_vcf: str,
    fasta_index: str,
    output_directory: str,
) -> tuple[str, str]:
    """
    Update VCF headers with contigs from FASTA index.

    Parameters
    ----------
    vcf_utils : VcfUtils
        VcfUtils instance for VCF operations
    cnmops_vcf : str
        Path to cn.mops VCF
    cnvpytor_vcf : str
        Path to CNVpytor VCF
    fasta_index : str
        Path to FASTA index file
    output_directory : str
        Output directory for temporary files

    Returns
    -------
    tuple[str, str]
        Paths to updated cn.mops and CNVpytor VCF files
    """
    logger.info("Updating VCF headers with contigs from FASTA index")
    cnmops_vcf_updated = pjoin(output_directory, "cnmops.updated_contigs.vcf.gz")
    cnvpytor_vcf_updated = pjoin(output_directory, "cnvpytor.updated_contigs.vcf.gz")

    vcf_utils.update_vcf_contigs_from_fai(cnmops_vcf, cnmops_vcf_updated, fasta_index)
    vcf_utils.update_vcf_contigs_from_fai(cnvpytor_vcf, cnvpytor_vcf_updated, fasta_index)

    return cnmops_vcf_updated, cnvpytor_vcf_updated


def _write_vcf_records_with_source(
    vcf_in: pysam.VariantFile,
    vcf_out: pysam.VariantFile,
    combined_header: pysam.VariantHeader,
    source_name: str,
) -> None:
    """
    Write VCF records to output file with CNV_SOURCE annotation.

    Note: This function clears all FILTER values from input records since
    FILTER definitions are not included in the combined header.

    Parameters
    ----------
    vcf_in : pysam.VariantFile
        Input VCF file to read records from
    vcf_out : pysam.VariantFile
        Output VCF file to write records to
    combined_header : pysam.VariantHeader
        Combined header for creating new records
    source_name : str
        Source name to add to CNV_SOURCE INFO field
    """
    logger.info(f"Writing records from {source_name} VCF")
    for record in vcf_in:
        # Clear filters - we remove filters imposed by the previous pipelines
        record.filter.clear()
        # Create new record with combined header
        new_record = VcfUtils.copy_vcf_record(record, combined_header)
        # Add source tag if not already present
        if "CNV_SOURCE" not in new_record.info:
            new_record.info["CNV_SOURCE"] = (source_name,)
        vcf_out.write(new_record)


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
    os.makedirs(output_directory, exist_ok=True)

    vcf_utils = VcfUtils()

    # Step 1: Update headers to contain same contigs from FASTA index
    cnmops_vcf_updated, cnvpytor_vcf_updated = _update_vcf_contigs(
        vcf_utils, cnmops_vcf, cnvpytor_vcf, fasta_index, output_directory
    )

    # Step 2: Open updated VCF files, combine headers (excluding FILTER fields), and add CNV_SOURCE tag
    logger.info("Combining VCF headers and adding CNV_SOURCE INFO tag")
    with pysam.VariantFile(cnmops_vcf_updated) as vcf1, pysam.VariantFile(cnvpytor_vcf_updated) as vcf2:
        # Combine headers (excluding FILTER fields)
        combined_header = combine_vcf_headers_for_cnv(vcf1.header, vcf2.header)

        # Add INFO tag for source if not already present
        if "CNV_SOURCE" not in combined_header.info:
            combined_header.info.add(
                "CNV_SOURCE",
                number=".",
                type="String",
                description="The tool that called this CNV (cn.mops or cnvpytor)",
            )

        # Step 3: Write records from both VCF files
        logger.info("Writing records from both VCF files to temporary combined VCF")
        temp_combined_vcf = pjoin(output_directory, "temp_combined.vcf.gz")

        with pysam.VariantFile(temp_combined_vcf, "w", header=combined_header) as vcf_out:
            _write_vcf_records_with_source(vcf1, vcf_out, combined_header, "cn.mops")
            _write_vcf_records_with_source(vcf2, vcf_out, combined_header, "cnvpytor")

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
    """Adds CNMOPS_SHORT_DUPLICATION filter to the short duplications that cn.mops returns.
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
    vu.collapse_vcf(duplication_vcf, collapsed_duplication_vcf, ignore_type=False, refdist=distance_threshold)
    vu.index_vcf(collapsed_duplication_vcf)
    temporary_files.append(collapsed_duplication_vcf)

    combined_calls = pjoin(output_dir, "temp_combined_calls.vcf.gz")
    vu.concat_vcf([deletion_vcf, collapsed_duplication_vcf], combined_calls)
    vu.index_vcf(combined_calls)
    temporary_files.append(combined_calls)

    with pysam.VariantFile(combined_calls) as vcf_in:
        hdr = vcf_in.header
        hdr.filters.add(
            "CNMOPS_SHORT_DUPLICATION",
            None,
            None,
            "Duplication length is shorter than the defined threshold in cn.mops calls.",
        )
        with pysam.VariantFile(combined_calls_annotated, "w", header=vcf_in.header) as vcf_out:
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


if __name__ == "__main__":
    main()
