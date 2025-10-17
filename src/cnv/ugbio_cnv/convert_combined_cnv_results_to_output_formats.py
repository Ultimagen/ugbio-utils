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
    "CNV_calls_source": (
        "CNV_SOURCE",
        1,
        "String",
        "the tool called this CNV. can be combination of: cn.mops, cnvpytor, gridss",
        "INFO",
    ),
    "jalign_written": ("JUMP_ALIGNMENTS", 1, "Float", "Number of jump alignments supporting this CNV", "INFO"),
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
    JALIGN_FILTER_VALUE: ("NO_JUMP_ALIGNMENT", None, None, "No jump alignment support", "FILTER"),
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

    # Add FORMAT
    header.add_line('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">')

    return header


def read_cnv_annotated_file_to_df(cnv_annotated_bed_file: str) -> pd.DataFrame:
    """
    Read a BED file and return a DataFrame.
    Args:
        cnv_annotated_bed_file (str): Path to the input BED file.
    Returns:
        pd.DataFrame: DataFrame containing the CNV data from the BED file.
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


def to_bed_name(row: pd.Series) -> str:
    """
    Convert a DataFrame row to a BED name string.
    Args:
        row (pd.Series): A row from the DataFrame.
    Returns:
        str: A string representing the BED name column.
    """
    filter_str = f"FILTER={row['filter']}"
    info_str = ""
    for info_tag in list(INFO_TAG_REGISTRY.keys()):
        if info_tag in row and pd.notna(row[info_tag]):
            info_str += f"{INFO_TAG_REGISTRY[info_tag][0]}={row[info_tag]};"
    for info_tag in ["CopyNumber", "RoundedCopyNumber", "SVLEN", "SVTYPE"]:
        if info_tag in row and pd.notna(row[info_tag]):
            info_str += f"{info_tag}={row[info_tag]};"
    return ";".join((info_str.strip(";"), filter_str.strip(";"))).strip(";")


def process_filter_columns(row: pd.Series, filter_tags_registry: dict = FILTER_TAG_REGISTRY) -> str:
    """
    Process filter columns for a single row, handling multiple filter values separated by | or ,.

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
            # Handle multiple values separated by | or ,
            value = value.replace("|", ";")
            parts = [part for part in value.split(";") if part and part not in ("PASS", ".")]
            filter_values.extend(parts)

    # Remove duplicates while preserving order
    unique_filters = list(set(filter_values))
    if len([x for x in unique_filters if x not in filter_tags_registry.keys()]):
        raise ValueError(
            f"Unknown filter values found: {[ x for x in unique_filters if x not in filter_tags_registry.keys() ]}"
        )

    unique_filters = [x for x in unique_filters if x in filter_tags_registry.keys()]

    # Return 'PASS' if no filters, otherwise return comma-separated filter list
    return "PASS" if len(unique_filters) == 0 else ",".join(sorted(unique_filters))


def write_combined_bed(outfile: str, cnv_annotated_bed_file: str) -> pd.DataFrame:
    """
    Write CNV calls from a BED file to another BED file.
    Args:
        outfile (str): Path to the output BED file.
        cnv_annotated_bed_file (str): Path to the input BED file containing combined (cn.mops+cnvpytor) CNV calls
            and annotated with UG-CNV-LCR.
    Returns:
        pd.DataFrame: DataFrame containing the CNV data written to the BED file.
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
    df_cnvs["name"] = df_cnvs.apply(to_bed_name, axis=1)
    df_cnvs[["chr", "start", "end", "name"]].to_csv(outfile, sep="\t", index=False, header=False)
    return df_cnvs


def _parse_bed_name_fields(name: str) -> dict[str, str]:
    """
    Parse the BED name field into a dictionary of key-value pairs.

    Args:
        name (str): The name field from BED format containing semicolon-separated key=value pairs.

    Returns:
        dict[str, str]: Dictionary of parsed field names and values.
    """
    return dict([x.split("=") for x in name.split(";")])


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
    record.alts = (f'<{row["SVTYPE"]}>',)
    return record


def _add_filters_to_record(record: pysam.VariantRecord, fields: dict[str, str]) -> None:
    """
    Add filter information to a VCF record.

    Args:
        record (pysam.VariantRecord): VCF record to modify.
        fields (dict[str, str]): Parsed fields containing filter information.
    """
    filters = fields["FILTER"].split(",")
    for f in filters:
        record.filter.add(f)


def _add_info_fields_to_record(record: pysam.VariantRecord, fields: dict[str, str]) -> None:
    """
    Add INFO field information to a VCF record.

    Args:
        record (pysam.VariantRecord): VCF record to modify.
        fields (dict[str, str]): Parsed fields containing INFO information.
    """
    for info_field, _ in fields.items():
        if info_field in INFO_TAG_REGISTRY or info_field in [
            "CopyNumber",
            "RoundedCopyNumber",
            "SVLEN",
            "SVTYPE",
        ]:
            if INFO_TAG_REGISTRY.get(info_field, (None, None, None, None, None))[-1] == "INFO" or info_field in [
                "CopyNumber",
                "RoundedCopyNumber",
                "SVLEN",
                "SVTYPE",
            ]:
                if info_field in ["CopyNumber", "SVLEN"]:
                    record.info[info_field] = float(fields[info_field])
                elif info_field in ["RoundedCopyNumber"]:
                    record.info[info_field] = int(fields[info_field])
                else:
                    record.info[info_field] = fields[info_field]


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


def _add_genotype_to_record(record: pysam.VariantRecord, sample_name: str, fields: dict[str, str]) -> None:
    """
    Add genotype information to a VCF record.

    Args:
        record (pysam.VariantRecord): VCF record to modify.
        sample_name (str): Name of the sample.
        fields (dict[str, str]): Parsed fields containing copy number information.
    """
    copy_number_value = fields.get("CopyNumber", None)
    gt = _determine_genotype(copy_number_value)
    record.samples[sample_name]["GT"] = gt


def write_combined_vcf(outfile: str, bed_df: pd.DataFrame, sample_name: str, fasta_index_file: str) -> None:
    """
    Write CNV calls from a BED file to a VCF file.
    Args:
        outfile (str): Path to the output VCF file.
        bed_df (pd.DataFrame): Dataframe ready to be converted to the VCF format (prepared when writing BED).
        sample_name (str): The name of the sample.
        fasta_index_file (str): Path to the reference genome index file (.fai).
    Returns:
        None
    """
    header = add_vcf_header(sample_name, fasta_index_file)

    with pysam.VariantFile(outfile, mode="w", header=header) as vcf_out:
        for _, row in bed_df.iterrows():
            # Parse BED name field into structured data
            fields = _parse_bed_name_fields(row["name"])

            # Create base VCF record with basic information
            record = _create_base_vcf_record(vcf_out, row)

            # Add filter information to the record
            _add_filters_to_record(record, fields)

            # Remove FILTER from fields to avoid duplication in INFO section
            del fields["FILTER"]

            # Add INFO field information to the record
            _add_info_fields_to_record(record, fields)

            # Add genotype information to the record
            _add_genotype_to_record(record, sample_name, fields)

            # Write the completed record to the VCF file
            vcf_out.write(record)


def run(argv):
    """
    converts combined CNV calls (from cnmops, cnvpytor, gridss) in bed format to vcf.
    input arguments:
    --cnv_annotated_bed_file: input bed file holding CNV calls.
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
        prog="convert_combined_cnv_results_to_output_formatis.py", description="converts CNV calls to bed and vcf."
    )

    parser.add_argument("--cnv_annotated_bed_file", help="input bed file holding CNV calls", required=True, type=str)
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

    # Open a VCF file for writing
    if args.out_directory:
        out_directory = args.out_directory
    else:
        out_directory = ""
    out_vcf_file = pjoin(out_directory, args.sample_name + ".cnv.vcf.gz")
    out_bed_file = pjoin(out_directory, args.sample_name + ".cnv.bed")
    vcf_ready_df = write_combined_bed(out_bed_file, args.cnv_annotated_bed_file)
    write_combined_vcf(out_vcf_file, vcf_ready_df, args.sample_name, args.fasta_index_file)

    # index outfile
    try:
        cmd = ["bcftools", "index", "-t", out_vcf_file]
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        logging.error(f"bcftools index command failed with exit code: {e.returncode}")
        sys.exit(1)  # Exit with error status

    logger.info(f"output file: {out_vcf_file}")
    logger.info(f"output file index: {out_vcf_file}.tbi")
    logger.info(f"output file: {out_bed_file}")

    return out_vcf_file, out_bed_file


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
