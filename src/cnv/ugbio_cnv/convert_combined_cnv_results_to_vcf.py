import argparse
import logging
import statistics
import subprocess
import sys
import warnings
from os.path import join as pjoin

import pandas as pd
import pysam
from ugbio_core.logger import logger

warnings.filterwarnings("ignore")


def add_vcf_header(sample_name: str, fasta_index_file: str, filter_tags: list[str]) -> pysam.VariantHeader:
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
    header.add_line('##ALT=<ID=<CNV>,Description="Copy number variant region">')
    header.add_line('##ALT=<ID=<DEL>,Description="Deletion relative to the reference">')
    header.add_line('##ALT=<ID=<DUP>,Description="Region of elevated copy number relative to the reference">')

    # Add FILTER
    header.add_line('##FILTER=<ID=PASS,Description="high confidence CNV call">')
    for filter_tag in filter_tags:
        header.add_line(f'##FILTER=<ID={filter_tag},Description="CNV calls filtered by {filter_tag}">')

    # Add INFO
    header.add_line(
        '##INFO=<ID=CopyNumber,Number=1,Type=Float,Description="average copy number detected from cn.mops">'
    )
    header.add_line(
        '##INFO=<ID=RoundedCopyNumber,Number=1,Type=Integer,Description="rounded copy number detected from cn.mops">'
    )
    header.add_line('##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="CNV length">')
    header.add_line('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="CNV type. can be DUP or DEL">')
    header.add_line(
        '##INFO=<ID=CNV_SOURCE,Number=1,Type=String,Description="the tool called this CNV. \
        can be combination of: cn.mops, cnvpytor, gridss">'
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
    df_cnvs = pd.read_csv(cnv_annotated_bed_file, sep="\t", header=None)
    base_columns = ["chr", "start", "end", "CNV_type", "CNV_calls_source", "copy_number"]
    if df_cnvs.shape[1] == len(base_columns) + 1:
        df_cnvs.columns = base_columns + ["filter"]
    elif df_cnvs.shape[1] == len(base_columns):
        df_cnvs.columns = base_columns
        df_cnvs["filter"] = "."
    else:
        raise ValueError("Unexpected number of columns in the TSV file.")
    return df_cnvs


def write_combined_vcf(outfile: str, cnv_annotated_bed_file: str, sample_name: str, fasta_index_file: str) -> None:
    """
    Write CNV calls from a BED file to a VCF file.
    Args:
        outfile (str): Path to the output VCF file.
        header (pysam.VariantHeader): The VCF header.
        cnv_annotated_bed_file (str): Path to the input BED file containing combined (cn.mops+cnvpytor) CNV calls
            and annotated with UG-CNV-LCR.
        sample_name (str): The name of the sample.
    """
    df_cnvs = read_cnv_annotated_file_to_df(cnv_annotated_bed_file)
    filter_tags = df_cnvs["filter"].unique().tolist()
    filter_tags = [tag for tag in filter_tags if tag != "."]
    header = add_vcf_header(sample_name, fasta_index_file, filter_tags)

    with pysam.VariantFile(outfile, mode="w", header=header) as vcf_out:
        for _, row in df_cnvs.iterrows():
            # Create a new VCF record
            chr_id = row["chr"]
            start = row["start"]
            end = row["end"]
            cnv_type = row["CNV_type"]
            cnv_call_source = row["CNV_calls_source"]
            copy_number = row["copy_number"]
            filter_val = row["filter"]

            if isinstance(copy_number, str):
                cn_list = copy_number.split(",")
                cn_list_filtered = [float(item) for item in cn_list if item not in (["DUP", "DEL"])]

                copy_number_value = cn_list[0]
                if len(cn_list_filtered) > 0:
                    copy_number_value = statistics.mean(cn_list_filtered)
            else:
                copy_number_value = copy_number

            cnv_type_value = f"<{cnv_type}>"

            filter_value_to_write = filter_val if filter_val != "." else ""

            record = vcf_out.new_record()
            record.contig = str(chr_id)
            record.start = start
            record.stop = end
            record.ref = "N"
            record.alts = (cnv_type_value,)
            if filter_value_to_write != "":
                record.filter.add(filter_value_to_write)
            else:
                record.filter.add("PASS")
            if not isinstance(copy_number_value, str):
                record.info["CopyNumber"] = float(copy_number_value)
                record.info["RoundedCopyNumber"] = int(round(float(copy_number_value)))
            record.info["SVLEN"] = int(end) - int(start)
            record.info["SVTYPE"] = cnv_type
            record.info["CNV_SOURCE"] = cnv_call_source
            # END position is automatically generated for multi-base variants

            # Set genotype information for each sample
            gt = [None, 1]
            if isinstance(copy_number_value, float):
                if int(round(copy_number_value)) == 1:
                    gt = [0, 1]
                elif int(round(copy_number_value)) == 0:
                    gt = [1, 1]
            record.samples[sample_name]["GT"] = (gt[0], gt[1])

            # Write the record to the VCF file
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
        prog="convert_combined_cnv_results_to_vcf.py", description="converts CNV calls in bed format to vcf."
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

    # header = add_vcf_header(args.sample_name, args.fasta_index_file)

    # Open a VCF file for writing
    if args.out_directory:
        out_directory = args.out_directory
    else:
        out_directory = ""
    outfile = pjoin(out_directory, args.sample_name + ".cnv.vcf.gz")
    write_combined_vcf(outfile, args.cnv_annotated_bed_file, args.sample_name, args.fasta_index_file)

    # index outfile
    try:
        cmd = ["bcftools", "index", "-t", outfile]
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        logging.error(f"bcftools index command failed with exit code: {e.returncode}")
        sys.exit(1)  # Exit with error status

    logger.info(f"output file: {outfile}")
    logger.info(f"output file index: {outfile}.tbi")
    return outfile


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
