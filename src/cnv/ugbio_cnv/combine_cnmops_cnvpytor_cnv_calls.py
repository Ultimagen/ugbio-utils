import argparse
import logging
import os
import re
import subprocess
import sys
from os.path import join as pjoin

import pandas as pd
import pysam
import ugbio_cnv.convert_combined_cnv_results_to_output_formats as output_results
import ugbio_core.misc_utils as mu
from pyfaidx import Fasta
from ugbio_core.logger import logger
from ugbio_core.vcf_utils import VcfUtils

bedmap = "bedmap"


def run_cmd(cmd):
    logger.info(cmd)
    subprocess.run(cmd, shell=True, check=True)  # noqa: S602


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="combine_cnmops_cnvpytor_cnv_calls.py",
        description="Combines CNV calls from various sources, filters, converts into final bed/VCF formats.",
    )

    parser.add_argument("--cnmops_cnv_calls", help="input bed file holding cn.mops CNV calls", required=True, type=str)
    parser.add_argument(
        "--cnvpytor_cnv_calls", help="input bed file holding cnvpytor CNV calls", required=True, type=str
    )
    parser.add_argument(
        "--del_jalign_merged_results", help="jaign results for Deletions in tsv format", required=True, type=str
    )
    parser.add_argument(
        "--distance_threshold",
        help="distance threshold for merging CNV segments",
        required=False,
        type=int,
        default=1500,
    )
    parser.add_argument(
        "--duplication_length_cutoff_for_cnmops_filter",
        help="Defines the minimum duplication length considered valid during cn.mops CNV filtering",
        required=False,
        type=int,
        default=10000,
    )
    parser.add_argument("--ug_cnv_lcr", help="UG-CNV-LCR bed file", required=False, type=str)
    parser.add_argument("--ref_fasta", help="reference genome fasta file", required=True, type=str)
    parser.add_argument("--fasta_index", help="fasta.fai file", required=True, type=str)
    parser.add_argument("--out_directory", help="output directory", required=False, type=str)
    parser.add_argument("--sample_name", help="sample name", required=True, type=str)
    parser.add_argument("--verbosity", help="Verbosity: ERROR, WARNING, INFO, DEBUG", required=False, default="INFO")

    return parser.parse_args(argv[1:])


def calculate_gaps_count_per_cnv(df_cnmops_calls: pd.DataFrame, ref_fasta: str) -> pd.DataFrame:
    """
    Calculate the number of 'N' bases in each CNV call region from the reference genome FASTA file.

     Parameters
     ----------
     df_cnmops_calls : pandas.DataFrame
         DataFrame containing CNV calls with columns ['chrom', 'start', 'end'].
     ref_fasta : str
         Path to the reference genome FASTA file.

     Returns
     -------
     pandas.DataFrame
         Updated DataFrame with additional columns: 'N_count', 'len', and 'pN'.
    """
    if not os.path.exists(ref_fasta):
        raise FileNotFoundError(f"Fasta file {ref_fasta} does not exist.")

    genome = Fasta(ref_fasta, build_index=False, rebuild=False)

    n_count = []
    for index, row in df_cnmops_calls.iterrows():  # noqa: B007
        chrom = row["chrom"]
        start = row["start"]
        end = row["end"]
        # pyfaidx uses 0-based start, end-exclusive indexing
        try:
            seq_obj = genome[chrom][start - 1 : end]  # Convert to 0-based
            if seq_obj is not None:
                seq = seq_obj.seq
                cnv_n_count = seq.upper().count("N")
            else:
                logger.warning(f"Could not retrieve sequence for {chrom}:{start}-{end}")
                cnv_n_count = 0
        except (KeyError, IndexError) as e:
            logger.warning(f"Error retrieving sequence for {chrom}:{start}-{end}: {e}")
            cnv_n_count = 0
        n_count.append(cnv_n_count)

    df_cnmops_calls["N_count"] = n_count
    df_cnmops_calls["len"] = df_cnmops_calls["end"] - df_cnmops_calls["start"] + 1
    df_cnmops_calls["pN"] = df_cnmops_calls["N_count"] / df_cnmops_calls["len"]

    return df_cnmops_calls


def parse_cnmops_cnv_calls(cnmops_cnv_calls: str, out_directory: str, ref_fasta: str, pN: float = 0) -> pd.DataFrame:  # noqa: N803
    """
    Parses cn.mops CNV calls from an input BED file.

    Parameters
    ----------
    cnmops_cnv_calls : str
        Path to the cn.mops CNV calls BED file.
    out_directory : str
        Output directory to store results.
    pN : float
        Threshold for filtering CNV calls based on the fraction of reference genome
        gaps (Ns) in the call region.

    Returns
    -------
    pd.DataFrame
        DataFrame containing parsed and filtered CNV calls.

    """
    cnmops_cnv_calls_tmp_file = f"{pjoin(out_directory, os.path.basename(cnmops_cnv_calls))}.tmp"

    # remove all tags from cnmops cnv calls file:
    run_cmd(
        f"cat {cnmops_cnv_calls} | sed 's/UG-CNV-LCR//g' | sed 's/LEN//g' | sed 's/|//g' \
            > {cnmops_cnv_calls_tmp_file}"
    )

    df_cnmops_cnvs = pd.read_csv(cnmops_cnv_calls_tmp_file, sep="\t", header=None)
    df_cnmops_cnvs.columns = ["chrom", "start", "end", "CN"]
    df_cnmops_cnvs = calculate_gaps_count_per_cnv(df_cnmops_cnvs, ref_fasta)
    # Filter by pN value
    df_cnmops_cnvs = df_cnmops_cnvs[df_cnmops_cnvs["pN"] <= pN]

    return df_cnmops_cnvs


# Helper function to check and add metadata records
def _add_metadata_records(record_dict1, record_dict2, record_type, enforced_info_specs, combined_header):
    """Add metadata records (INFO/FORMAT/FILTER) to combined header."""
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


def combine_vcf_headers_for_cnv(header1: pysam.VariantHeader, header2: pysam.VariantHeader) -> pysam.VariantHeader:
    """
    Combine two VCF headers into a single header for CNV/SV variant calls.

    This function is specifically designed for merging headers from different CNV/SV
    callers (e.g., cn.mops and CNVpytor). It creates a new header that includes all
    metadata from both input headers with special handling for structural variant fields.

    For INFO and FORMAT fields with the same ID:
    - If type and number match, use the first definition
    - If type or number differ, raise RuntimeError

    FILTER fields are not included in the combined header.

    Special enforcement for CNV/SV-related fields to ensure VCF spec compliance:
    - SVLEN: enforces Number="." (variable-length array) per VCF 4.2 spec for structural variants
    - SVTYPE: enforces Number=1 (single value) per VCF 4.2 spec for structural variants

    Parameters
    ----------
    header1 : pysam.VariantHeader
        The first VCF header (takes precedence in case of identical collisions)
    header2 : pysam.VariantHeader
        The second VCF header to merge

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
        "SVLEN": (".", "Integer", "CNV length"),
        "SVTYPE": (1, "String", "CNV type. can be DUP or DEL"),
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
        # Clear filters before copying to avoid KeyError with undefined filters
        record.filter.clear()
        # Create new record with combined header
        new_record = VcfUtils.copy_vcf_record(record, combined_header)
        # Add source tag if not already present
        if "CNV_SOURCE" not in new_record.info:
            new_record.info["CNV_SOURCE"] = (source_name,)
        vcf_out.write(new_record)


def _cleanup_temp_files(temp_files: list[str]) -> None:
    """
    Remove temporary files and their indices.

    Parameters
    ----------
    temp_files : list[str]
        List of temporary file paths to remove
    """
    logger.info("Cleaning up temporary files")
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
        # Also remove index files
        for ext in [".tbi", ".csi"]:
            index_file = temp_file + ext
            if os.path.exists(index_file):
                os.unlink(index_file)


def combine_cnv_vcfs(
    cnmops_vcf: str,
    cnvpytor_vcf: str,
    fasta_index: str,
    output_vcf: str,
    output_directory: str,
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
    output_directory : str
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


def get_dup_cnmops_cnv_calls(
    df_cnmops: pd.DataFrame, sample_name: str, out_directory: str, distance_threshold: int
) -> str:
    """
    Parameters
    ----------
    df_cnmops : pandas.DataFrame
        DataFrame holding cn.mops CNV calls.
    sample_name : str
        Sample name.
    out_directory : str
        Output folder to store results.
    distance_threshold : int
        Distance threshold for merging CNV segments.

    Returns
    -------
    str
        Path to the duplications called by cn.mops bed file.
    """
    # get duplications from cn.mops calls
    cnmops_cnvs_dup = pjoin(out_directory, f"{sample_name}.cnmops_cnvs.DUP.bed")

    # df_cnmops = pd.read_csv(cnmops_cnv_calls, sep="\t", header=None)
    # df_cnmops.columns = ["chrom", "start", "end", "CN"]
    def extract_cn_number(item):
        match = re.search(r"CN([\d\.]+)", item)
        return match.group(1) if match else 0

    df_cnmops["cn_numbers"] = [extract_cn_number(item) for item in df_cnmops["CN"]]
    out_cnmops_cnvs_dup_calls = pjoin(out_directory, f"{sample_name}.cnmops_cnvs.DUP.calls.bed")
    neutral_cn = 2
    df_cnmops[df_cnmops["cn_numbers"].astype(float) > neutral_cn][["chrom", "start", "end", "CN"]].to_csv(
        out_cnmops_cnvs_dup_calls, sep="\t", header=False, index=False
    )

    if os.path.getsize(out_cnmops_cnvs_dup_calls) > 0:
        run_cmd(
            f"cat {out_cnmops_cnvs_dup_calls} | \
                bedtools sort -i - | \
                bedtools merge -d {distance_threshold} -c 4 -o distinct -i - | \
                awk '$3-$2>=10000' | \
                sed 's/$/\\tDUP\\tcn.mops/' | \
                cut -f1,2,3,5,6,4 > {cnmops_cnvs_dup}"
        )

        df_cnmops_cnvs_dup = pd.read_csv(cnmops_cnvs_dup, sep="\t", header=None)
        df_cnmops_cnvs_dup.columns = ["chrom", "start", "end", "CN", "CNV_type", "source"]
        df_cnmops_cnvs_dup["copy_number"] = df_cnmops_cnvs_dup["CN"].apply(lambda x: x.replace("CN", ""))

        out_cnmops_cnvs_dup = pjoin(out_directory, f"{sample_name}.cnmops_cnvs.DUP.all_fields.bed")
        df_cnmops_cnvs_dup[["chrom", "start", "end", "CNV_type", "source", "copy_number"]].to_csv(
            out_cnmops_cnvs_dup, sep="\t", header=False, index=False
        )

        return out_cnmops_cnvs_dup
    else:
        logger.info("No duplications found in cn.mops CNV calls.")
        return ""


def parse_cnvpytor_cnv_calls(cnvpytor_cnv_calls: str, pN: float = 0) -> pd.DataFrame:  # noqa: N803
    """
    Parses cnvpytor CNV calls from a tsv file.

    Parameters
    ----------
    cnvpytor_cnv_calls : str
        Path to the cnvpytor CNV calls bed file.
    pN : float
        Threshold for filtering CNV calls based on the fraction of reference genome
        gaps (Ns) in call region.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing parsed CNV calls.
    """
    # Result is stored in tab separated files with following columns:
    # CNV type: "deletion" or "duplication",
    # CNV region (chr:start-end),
    # CNV size,
    # CNV level - read depth normalized to 1,
    # e-val1 -- e-value (p-value multiplied by genome size divided by bin size) calculated
    #           using t-test statistics between RD statistics in the region and global,
    # e-val2 -- e-value (p-value multiplied by genome size divided by bin size) from the probability of RD values within
    #           the region to be in the tails of a gaussian distribution of binned RD,
    # e-val3 -- same as e-val1 but for the middle of CNV,
    # e-val4 -- same as e-val2 but for the middle of CNV,
    # q0 -- fraction of reads mapped with q0 quality in call region,
    # pN -- fraction of reference genome gaps (Ns) in call region,
    # dG -- distance from closest large (>100bp) gap in reference genome.

    df_cnvpytor_cnvs = pd.read_csv(cnvpytor_cnv_calls, sep="\t", header=None)
    df_cnvpytor_cnvs.columns = [
        "cnv_type",
        "cnv_region",
        "len",
        "cnv_level",
        "e-val1",
        "e-val2",
        "e-val3",
        "e-val4",
        "q0",
        "pN",
        "dG",
    ]

    # Split cnv_region into 'chr', 'start', 'end'
    df_cnvpytor_cnvs[["chrom", "pos"]] = df_cnvpytor_cnvs["cnv_region"].str.split(":", expand=True)
    df_cnvpytor_cnvs[["start", "end"]] = df_cnvpytor_cnvs["pos"].str.split("-", expand=True)
    df_cnvpytor_cnvs = df_cnvpytor_cnvs.drop(columns="pos")
    df_cnvpytor_cnvs["start"] = df_cnvpytor_cnvs["start"].astype(int)
    df_cnvpytor_cnvs["end"] = df_cnvpytor_cnvs["end"].astype(int)

    # Filter by pN value
    df_cnvpytor_cnvs = df_cnvpytor_cnvs[df_cnvpytor_cnvs["pN"] <= 0]

    return df_cnvpytor_cnvs


def get_dup_cnvpytor_cnv_calls(df_cnvpytor_cnv_calls: pd.DataFrame, sample_name: str, out_directory: str) -> str:  # noqa: N803
    """
    Parameters
    ----------
    df_cnvpytor_cnv_calls : pandas.DataFrame
        DataFrame holding cnvpytor CNV calls.
    sample_name : str
        Sample name.
    out_directory : str
        Output folder to store results.

    Returns
    -------
    str
        Path to the duplications called by cnvpytor bed file.
    """
    cnvpytor_cnvs_dup = pjoin(out_directory, f"{sample_name}.cnvpytor_cnvs.DUP.bed")
    df_cnvpytor_cnv_calls_duplications = df_cnvpytor_cnv_calls[
        df_cnvpytor_cnv_calls["cnv_type"] == "duplication"
    ].copy()
    df_cnvpytor_cnv_calls_duplications["cnv_type"] = "DUP"
    df_cnvpytor_cnv_calls_duplications["copy_number"] = "DUP"
    df_cnvpytor_cnv_calls_duplications["source"] = "cnvpytor"

    if len(df_cnvpytor_cnv_calls_duplications) > 0:
        df_cnvpytor_cnv_calls_duplications[["chrom", "start", "end", "cnv_type", "source", "copy_number"]].to_csv(
            cnvpytor_cnvs_dup, sep="\t", header=False, index=False
        )
        return cnvpytor_cnvs_dup
    else:
        logger.info("No duplications found in cnvpytor CNV calls.")
        return ""


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

    df_cnmops_cnvpytor_del = calculate_gaps_count_per_cnv(df_cnmops_cnvpytor_del, ref_fasta)
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


def run(argv):
    """
    Combine CNVs from cn.mops and cnvpytor using jalign results and convert them to VCF.

    Parameters
    ----------
    argv : list of str
        Command-line arguments.

    Input Arguments
    ---------------
    --cnmops_cnv_calls : str
        Input BED file holding cn.mops CNV calls.
    --cnvpytor_cnv_calls : str
        Input BED file holding cnvpytor CNV calls.
    --del_jalign_merged_results : str
        Jalign results for deletions in TSV format.

    Output Files
    ------------
    <sample_name>.cnmops_cnvpytor.cnvs.combined.bed : str
        Combined CNV calls called by cn.mops and cnvpytor.
    <sample_name>.cnmops_cnvpytor.cnvs.combined.UG-CNV-LCR_annotate.bed : str
        Combined CNV calls with UG-CNV-LCR annotation.
    """
    args = __parse_args(argv)
    logger.setLevel(getattr(logging, args.verbosity))

    out_directory = args.out_directory
    sample_name = args.sample_name
    # format cnvpytor cnv calls :
    df_cnmops_cnv_calls = parse_cnmops_cnv_calls(args.cnmops_cnv_calls, out_directory, args.ref_fasta)
    df_cnvpytor_cnv_calls = parse_cnvpytor_cnv_calls(args.cnvpytor_cnv_calls)

    ############################
    ### process DUPlications ###
    ############################
    out_cnmops_cnvs_dup = get_dup_cnmops_cnv_calls(
        df_cnmops_cnv_calls, sample_name, out_directory, args.distance_threshold
    )
    out_cnvpytor_cnvs_dup = get_dup_cnvpytor_cnv_calls(df_cnvpytor_cnv_calls, sample_name, out_directory)
    # merge duplications
    if not out_cnmops_cnvs_dup and not out_cnvpytor_cnvs_dup:
        logger.info("No duplications found in cn.mops and cnvpytor CNV calls.")
        cnmops_cnvpytor_merged_dup = ""
    else:
        cnmops_cnvpytor_merged_dup = pjoin(out_directory, f"{sample_name}.cnmops_cnvpytor.DUP.merged.bed")
        run_cmd(
            f'cat {out_cnmops_cnvs_dup} {out_cnvpytor_cnvs_dup} | bedtools sort -i - | \
            bedtools merge -c 4,5,6 -o distinct -i - | \
                awk \'{{print $0 "\t" "0" }}\' > {cnmops_cnvpytor_merged_dup}'
        )

    ############################
    ###  process DELetions   ###
    ############################

    out_del_jalign_merged = process_del_jalign_results(
        args.del_jalign_merged_results, sample_name, out_directory, ref_fasta=args.ref_fasta, pN=0
    )

    # combine results
    out_cnvs_combined = pjoin(out_directory, f"{sample_name}.cnmops_cnvpytor.cnvs.combined.bed")
    columns = "\t".join(["#chr", "start", "end", "CNV_type", "CNV_calls_source", "copy_number", "jalign_written"])
    with open(out_cnvs_combined, "w") as out_fh:
        out_fh.write(f"{columns}\n")

    # Use unix sort instead of bedtools sort for proper end coordinate sorting when starts are identical
    run_cmd(
        f"cat {cnmops_cnvpytor_merged_dup} {out_del_jalign_merged} | sort -k1,1 -k2,2n -k3,3n >> {out_cnvs_combined}"
    )
    logger.info(f"out_cnvs_combined: {out_cnvs_combined}")

    if args.ug_cnv_lcr:
        # annotate with ug-cnv-lcr if provided
        # result file should be in the following format:
        # ["chr", "start", "end", "CNV_type", "CNV_calls_source", "copy_number", "UG-CNV-LCR"]

        sorted_ug_cnv_lcr = f"{out_cnvs_combined}.lcr.bed"
        run_cmd(f"sort -k1,1 -k2,2n -k3,3n {args.ug_cnv_lcr} > {sorted_ug_cnv_lcr}")

        out_cnvs_combined_annotated = f"{out_cnvs_combined}.annotate.tsv"
        run_cmd(
            f"bedmap --header --echo --echo-map-id-uniq --delim '\\t' --bases-uniq-f \
            {out_cnvs_combined} {sorted_ug_cnv_lcr} > {out_cnvs_combined_annotated}"
        )

        os.unlink(sorted_ug_cnv_lcr)

        logger.info(f"out_cnvs_combined_annotated: {out_cnvs_combined_annotated}")

        overlap_filtration_cutoff = 0.5  # 50% overlap with LCR regions
        # note: despite what it sounds, bedmap --header skips the header in the output
        df_annotate_calls = pd.read_csv(out_cnvs_combined_annotated, sep="\t", header=None)
        df_annotate_calls.columns = [
            "#chr",
            "start",
            "end",
            "CNV_type",
            "CNV_calls_source",
            "copy_number",
            "jalign_written",
            "UG-CNV-LCR",
            "pUG-CNV-LCR_overlap",
        ]
        df_annotate_calls["LCR_label_value"] = df_annotate_calls.apply(
            lambda row: row["UG-CNV-LCR"] if row["pUG-CNV-LCR_overlap"] >= overlap_filtration_cutoff else ".", axis=1
        )

        df_annotate_calls[
            [
                "#chr",
                "start",
                "end",
                "CNV_type",
                "CNV_calls_source",
                "copy_number",
                "LCR_label_value",
                "jalign_written",
            ]
        ].to_csv(out_cnvs_combined_annotated, sep="\t", index=False)
        logger.info(f"out_cnvs_combined_annotated: {out_cnvs_combined_annotated}")

    else:
        out_cnvs_combined_annotated = out_cnvs_combined

    # convert to vcf
    vcf_args = [
        "combine_cnmops_cnvpytor_cnv_calls",
        "--cnv_annotated_bed_file",
        out_cnvs_combined_annotated,
        "--fasta_index_file",
        args.fasta_index,
        "--out_directory",
        out_directory,
        "--sample_name",
        sample_name,
    ]
    logger.info(vcf_args)
    out_cnvs_combined_annotated_vcf = output_results.run(vcf_args)
    logger.info(f"out_cnvs_combined_annotated_vcf: {out_cnvs_combined_annotated_vcf}")


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
