import argparse
import logging
import statistics
import subprocess
import sys
import warnings
from os.path import join as pjoin

import numpy as np
import pandas as pd
import pysam
from ugbio_core.logger import logger

warnings.filterwarnings("ignore")

INFO_TAG_REGISTRY = {
    "CNV_SOURCE": (
        "CNV_SOURCE",
        1,
        "String",
        "the tool called this CNV. can be combination of: cn.mops, cnvpytor, gridss",
        "INFO",
    ),
    "JUMP_ALIGNMENTS": ("JUMP_ALIGNMENTS", 1, "Float", "Number of jump alignments supporting this CNV", "INFO"),
}

FILTER_COLUMNS_REGISTRY = ["jalign_filter", "LCR_label_value"]
JALIGN_FILTER_VALUE = "NO_JUMP_ALIGNMENT"
FILTER_TAG_REGISTRY = {
    "Clusters": ("Clusters", None, None, "Overlaps with locations with frequent clusters of CNV", "FILTER"),
    "Coverage-Mappability": (
        "Coverage-Mappability",
        None,
        None,
        "Overlaps with low coverage or low mappability regions",
        "FILTER",
    ),
    "Telomere_Centromere": (
        "Telomere_Centromere",
        None,
        None,
        "Overlaps with telomere or centromere regions",
        "FILTER",
    ),
    JALIGN_FILTER_VALUE: ("NO_JUMP_ALIGNMENT", None, None, "No jump alignment support (dels only)", "FILTER"),
}


def add_vcf_header(sample_name: str, fasta_index_file: str) -> pysam.VariantHeader:
    """
    Create a VCF header, for CNV calls, with the given sample name and reference genome information.
    Args:
        sample_name (str): The name of the sample.
        fasta_index_file (str): Path to the reference genome index file (.fai).
    Returns:
        pysam.VariantHeader: The VCF header with the specified sample name and reference genome information.
    """
    header = pysam.VariantHeader()

    # Add meta-information to the header
    header.add_meta("fileformat", value="VCFv4.2")
    header.add_meta("source", value="ULTIMA_CNV")

    # Add sample names to the header
    header.add_sample(sample_name)

    header.add_line("##VCF_TYPE=ULTIMA_CNV")

    # Add contigs info to the header
    df_genome = pd.read_csv(fasta_index_file, sep="\t", header=None, usecols=[0, 1])
    df_genome.columns = ["chr", "length"]
    for _, row in df_genome.iterrows():
        chr_id = row["chr"]
        length = row["length"]
        header.add_line(f"##contig=<ID={chr_id},length={length}>")

    # Add ALT
    header.add_line('##ALT=<ID=CNV,Description="Copy number variant region">')
    header.add_line('##ALT=<ID=DEL,Description="Deletion relative to the reference">')
    header.add_line('##ALT=<ID=DUP,Description="Region of elevated copy number relative to the reference">')

    # Add FILTER
    header.add_line('##FILTER=<ID=PASS,Description="High confidence CNV call">')
    for filter_tag in FILTER_TAG_REGISTRY.values():
        if filter_tag[-1] == "FILTER":
            header.add_line(f'##FILTER=<ID={filter_tag[0]},Description="{filter_tag[-2]}">')

    # Add INFO
    header.add_line(
        '##INFO=<ID=CopyNumber,Number=1,Type=Float,Description="average copy number detected from cn.mops">'
    )
    header.add_line(
        '##INFO=<ID=RoundedCopyNumber,Number=1,Type=Integer,Description="rounded copy number detected from cn.mops">'
    )
    header.add_line('##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="CNV length">')
    header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="CNV type. can be DUP or DEL">')
    for info_tag in INFO_TAG_REGISTRY.values():
        if info_tag[-1] == "INFO":
            header.add_line(
                f'##INFO=<ID={info_tag[0]},Number={info_tag[1]},Type={info_tag[2]},Description="{info_tag[3]}">'
            )

    # Add END field for structural variants
    header.add_line('##INFO=<ID=END,Number=1,Type=Integer,Description="Stop position of the interval">')

    # Add FORMAT
    header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')

    return header


def read_cnv_annotated_file_to_df(cnv_annotated_bed_file: str) -> pd.DataFrame:
    """
    Read an annotated CNV file and return a DataFrame.
    Args:
        cnv_annotated_bed_file (str): Path to the input annotated CNV file.
    Returns:
        pd.DataFrame: DataFrame containing the CNV data from the file.
    """
    df_cnvs = pd.read_csv(cnv_annotated_bed_file, sep="\t")
    df_cnvs = df_cnvs.rename(columns={"#chr": "chr"})
    return df_cnvs


def calculate_copy_number(copy_number: str | int) -> float | str | None:
    """
    Calculate the average copy number from a string of copy number values.
    Args:
        copy_number (str | int): A string containing copy number values separated by commas.
    Returns:
        float | str | None: The average copy number as a float, or the original string if it is a float or symbolic
    """
    if isinstance(copy_number, str):
        cn_list = copy_number.split(",")
        cn_list_filtered = [float(item) for item in cn_list if item not in (["DUP", "DEL"])]

        if len(cn_list_filtered) > 0:
            copy_number_value = statistics.mean(cn_list_filtered)
        else:
            copy_number_value = None
    else:
        copy_number_value = float(copy_number)
    return copy_number_value


def process_filter_columns(row: pd.Series, filter_tags_registry: dict = FILTER_TAG_REGISTRY) -> str:
    """
    Process filter columns for a single row, handling multiple filter values separated by | or ;.

    Args:
        row (pd.Series): A row from the DataFrame containing filter columns.
        filter_tags_registry (dict): A dictionary of valid filter tags.

    Returns:
        str: Comma-separated string of unique filter values, or 'PASS' if no filters apply.
    """
    filter_values = []

    for col in FILTER_COLUMNS_REGISTRY:
        if col in row and pd.notna(row[col]):
            value = str(row[col])
            # Handle multiple values separated by | or ;
            value = value.replace("|", ";")
            parts = [part for part in value.split(";") if part and part not in ("PASS", ".")]
            filter_values.extend(parts)

    # Remove duplicates while preserving order
    unique_filters = list(set(filter_values))
    if [x for x in unique_filters if x not in filter_tags_registry.keys()]:
        raise ValueError(
            f"Unknown filter values found: {[x for x in unique_filters if x not in filter_tags_registry.keys()]}"
        )

    unique_filters = [x for x in unique_filters if x in filter_tags_registry.keys()]

    # Return 'PASS' if no filters, otherwise return comma-separated filter list
    return "PASS" if len(unique_filters) == 0 else ",".join(sorted(unique_filters))


def prepare_cnv_dataframe(cnv_annotated_bed_file: str) -> pd.DataFrame:
    """
    Prepare CNV dataframe for VCF output by processing filters and calculating derived fields.
    Args:
        cnv_annotated_bed_file (str): Path to the input file containing combined CNV calls
            and annotated with UG-CNV-LCR.
    Returns:
        pd.DataFrame: DataFrame containing the processed CNV data ready for VCF conversion.
    """
    df_cnvs = read_cnv_annotated_file_to_df(cnv_annotated_bed_file)
    df_cnvs["filter"] = df_cnvs.apply(process_filter_columns, axis=1)

    df_cnvs["CopyNumber"] = df_cnvs["copy_number"].apply(calculate_copy_number).replace(np.nan, None)
    df_cnvs["RoundedCopyNumber"] = df_cnvs["CopyNumber"].apply(
        lambda x: int(round(x)) if isinstance(x, float) else pd.NA
    )
    df_cnvs["RoundedCopyNumber"] = df_cnvs["RoundedCopyNumber"].astype("Int64")
    df_cnvs["SVLEN"] = df_cnvs["end"] - df_cnvs["start"]
    df_cnvs["SVTYPE"] = df_cnvs["CNV_type"]
    return df_cnvs


def _create_base_vcf_record(vcf_out: pysam.VariantFile, row: pd.Series) -> pysam.VariantRecord:
    """
    Create a base VCF record with chromosome, position, and variant type information.

    Args:
        vcf_out (pysam.VariantFile): VCF file handle for creating new records.
        row (pd.Series): DataFrame row containing CNV information.

    Returns:
        pysam.VariantRecord: Base VCF record with basic information set.
    """
    record = vcf_out.new_record()
    record.contig = str(row["chr"])
    record.start = row["start"]
    record.stop = row["end"]
    record.ref = "N"
    record.alts = (f"<{row['SVTYPE']}>",)
    return record


def _add_filters_to_record(record: pysam.VariantRecord, filter_value: str) -> None:
    """
    Add filter information to a VCF record.

    Args:
        record (pysam.VariantRecord): VCF record to modify.
        filter_value (str): Filter value from the dataframe.
    """
    if filter_value == "PASS":
        record.filter.add("PASS")
    else:
        filters = filter_value.split(",")
        for f in filters:
            record.filter.add(f)


def _get_possible_column_names(original_col: str) -> list[str]:
    """Get possible column names for backward compatibility."""
    possible_cols = [original_col]
    if original_col == "CNV_SOURCE":
        possible_cols.append("CNV_calls_source")
    elif original_col == "JUMP_ALIGNMENTS":
        possible_cols.append("jalign_written")
    return possible_cols


def _add_registry_info_field(
    record: pysam.VariantRecord, row: pd.Series, vcf_tag: str, possible_cols: list[str], data_type: str
) -> None:
    """Add a single INFO field from the registry to the VCF record."""
    for col in possible_cols:
        if col in row and pd.notna(row[col]):
            if data_type == "Integer":
                record.info[vcf_tag] = int(row[col])
            elif data_type == "Float":
                record.info[vcf_tag] = float(row[col])
            else:
                record.info[vcf_tag] = str(row[col])
            break  # Only use the first available column


def _add_standard_info_fields(record: pysam.VariantRecord, row: pd.Series) -> None:
    """Add standard INFO fields to the VCF record."""
    if pd.notna(row.get("CopyNumber")):
        record.info["CopyNumber"] = float(row["CopyNumber"])
    if pd.notna(row.get("RoundedCopyNumber")):
        record.info["RoundedCopyNumber"] = int(row["RoundedCopyNumber"])
    if pd.notna(row.get("SVLEN")):
        record.info["SVLEN"] = int(row["SVLEN"])
    if pd.notna(row.get("SVTYPE")):
        record.info["SVTYPE"] = str(row["SVTYPE"])


def _add_info_fields_to_record(record: pysam.VariantRecord, row: pd.Series) -> None:
    """
    Add INFO field information to a VCF record directly from dataframe row.

    Args:
        record (pysam.VariantRecord): VCF record to modify.
        row (pd.Series): DataFrame row containing CNV information.
    """
    # Add INFO fields from registry first, handling both possible column names
    for original_col, (vcf_tag, _, data_type, _, _) in INFO_TAG_REGISTRY.items():
        possible_cols = _get_possible_column_names(original_col)
        _add_registry_info_field(record, row, vcf_tag, possible_cols, data_type)

    # Add standard INFO fields after registry fields
    _add_standard_info_fields(record, row)


def _determine_genotype(copy_number_value: str | None) -> tuple[int | None, int]:
    """
    Determine the genotype based on copy number value.

    Args:
        copy_number_value (str | None): String representation of copy number.

    Returns:
        tuple[int | None, int]: Genotype tuple for VCF format.
    """
    gt = [None, 1]
    if copy_number_value is not None:
        try:
            cn_float = float(copy_number_value)
            rounded_cn = int(round(cn_float))
            if rounded_cn == 1:
                gt = [0, 1]
            elif rounded_cn == 0:
                gt = [1, 1]
        except (ValueError, TypeError):
            pass  # Keep default gt
    return (gt[0], gt[1])


def _add_genotype_to_record(record: pysam.VariantRecord, sample_name: str, row: pd.Series) -> None:
    """
    Add genotype information to a VCF record.

    Args:
        record (pysam.VariantRecord): VCF record to modify.
        sample_name (str): Name of the sample.
        row (pd.Series): DataFrame row containing copy number information.
    """
    copy_number_value = row.get("CopyNumber", None)
    if pd.notna(copy_number_value):
        copy_number_value = str(copy_number_value)
    else:
        copy_number_value = None
    gt = _determine_genotype(copy_number_value)
    record.samples[sample_name]["GT"] = gt


def write_combined_vcf(outfile: str, cnv_df: pd.DataFrame, sample_name: str, fasta_index_file: str) -> None:
    """
    Write CNV calls directly from dataframe to a VCF file.
    Args:
        outfile (str): Path to the output VCF file.
        cnv_df (pd.DataFrame): Dataframe containing processed CNV data.
        sample_name (str): The name of the sample.
        fasta_index_file (str): Path to the reference genome index file (.fai).
    Returns:
        None
    """
    header = add_vcf_header(sample_name, fasta_index_file)

    with pysam.VariantFile(outfile, mode="w", header=header) as vcf_out:
        for _, row in cnv_df.iterrows():
            # Create base VCF record with basic information
            record = _create_base_vcf_record(vcf_out, row)

            # Add filter information to the record
            _add_filters_to_record(record, row["filter"])

            # Add INFO field information to the record
            _add_info_fields_to_record(record, row)

            # Add genotype information to the record
            _add_genotype_to_record(record, sample_name, row)

            # Write the completed record to the VCF file
            vcf_out.write(record)


def run(argv):
    """
    converts combined CNV calls (from cnmops, cnvpytor, gridss) and outputs VCF file.
    input arguments:
    --cnv_annotated_bed_file: input file holding CNV calls.
    --fasta_index_file: (.fai file) tab delimeted file holding reference genome chr ids with their lengths.
    --out_directory: output directory
    --sample_name: sample name
    output files:
    vcf file: <sample_name>.cnv.vcf.gz
        shows called CNVs in zipped vcf format.
    vcf index file: <sample_name>.cnv.vcf.gz.tbi
        vcf corresponding index file.
    """
    parser = argparse.ArgumentParser(
        prog="convert_combined_cnv_results_to_output_formats.py", description="converts CNV calls to VCF."
    )

    parser.add_argument("--cnv_annotated_bed_file", help="input file holding CNV calls", required=True, type=str)
    parser.add_argument(
        "--fasta_index_file",
        help="tab delimeted file holding reference genome chr ids with their lengths. (.fai file)",
        required=True,
        type=str,
    )
    parser.add_argument("--out_directory", help="output directory", required=False, type=str)
    parser.add_argument("--sample_name", help="sample name", required=True, type=str)
    parser.add_argument("--verbosity", help="Verbosity: ERROR, WARNING, INFO, DEBUG", required=False, default="INFO")

    args = parser.parse_args(argv[1:])
    logger.setLevel(getattr(logging, args.verbosity))

    # Prepare output file path
    if args.out_directory:
        out_directory = args.out_directory
    else:
        out_directory = ""
    out_vcf_file = pjoin(out_directory, args.sample_name + ".cnv.vcf.gz")

    # Process CNV data and write VCF
    cnv_df = prepare_cnv_dataframe(args.cnv_annotated_bed_file)
    write_combined_vcf(out_vcf_file, cnv_df, args.sample_name, args.fasta_index_file)

    # index outfile
    try:
        cmd = ["bcftools", "index", "-t", out_vcf_file]
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        logging.error(f"bcftools index command failed with exit code: {e.returncode}")
        sys.exit(1)  # Exit with error status

    logger.info(f"output file: {out_vcf_file}")
    logger.info(f"output file index: {out_vcf_file}.tbi")

    return out_vcf_file


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
