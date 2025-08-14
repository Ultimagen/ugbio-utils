import argparse
import logging
import os
import subprocess
import sys
import tempfile
from enum import Enum
from os.path import join as pjoin

import pandas as pd
from ugbio_core.logger import logger
from ugbio_core.vcfbed import vcftools

from ugbio_featuremap import featuremap_xgb_prediction


class DefaultCustomInfoFieldsWithSingleValue(Enum):
    READ_COUNT = ("X_READ_COUNT", "Number of reads containing this location", "Integer")
    FILTERED_COUNT = (
        "X_FILTERED_COUNT",
        "Number of reads containing this location that pass the adjacent base filter",
        "Integer",
    )
    TRINUC_CONTEXT_WITH_ALT = ("trinuc_context_with_alt", "reference trinucleotide context and alt base", "String")
    HMER_CONTEXT_ALT = (
        "hmer_context_alt",
        "homopolymer context in the ref allele assuming the variant considered only up to length 20",
        "Integer",
    )

    def __init__(self, label, description, type_):
        self.label = label
        self.description = description
        self.type = type_


class LocusFeaturesWithSingleValue(Enum):
    HMER_CONTEXT_REF = ("hmer_context_ref", "reference homopolymer context up to length 20", "Integer")
    PREV_1 = ("prev_1", "1 bases in the reference before variant", "String")
    PREV_2 = ("prev_2", "2 bases in the reference before variant", "String")
    PREV_3 = ("prev_3", "3 bases in the reference before variant", "String")
    NEXT_1 = ("next_1", "1 bases in the reference after variant", "String")
    NEXT_2 = ("next_2", "2 bases in the reference after variant", "String")
    NEXT_3 = ("next_3", "3 bases in the reference after variant", "String")
    IS_CYCLE_SKIP = ("is_cycle_skip", "True if the SNV is a cycle skip", "Integer")

    def __init__(self, label, description, type_):
        self.label = label
        self.description = description
        self.type = type_


def enum_to_dict(class_name):
    return {member.label: [member.description, member.type] for member in class_name}


def get_combined_vcf_features():
    """
    Returns the features used in the featuremap_xgb_prediction module.

    Returns
    -------
    dict
        Dictionary where keys are the feature tags and values are lists containing
        the description and type of the feature.
    """
    default_custom_info_fields_single_value = enum_to_dict(DefaultCustomInfoFieldsWithSingleValue)
    added_agg_features = featuremap_xgb_prediction.added_agg_features
    info_field_tags = {**default_custom_info_fields_single_value, **added_agg_features}
    lucos_info_fields = enum_to_dict(LocusFeaturesWithSingleValue)
    return info_field_tags, lucos_info_fields


def move_vcf_value_from_INFO_to_FORMAT(input_vcf, output_vcf, info_field_tags, lucos_info_fields):  # noqa: N802, PLR0915
    """
    Move the values from INFO to FORMAT in a VCF file.

    This is necessary for the XGB model to work correctly.

    Parameters
    ----------
    input_vcf : str
        Path to the input VCF file.
    output_vcf : str
        Path to the output VCF file.
    info_field_tags : dict
        Dictionary of INFO field tags to be moved to FORMAT.

    Returns
    -------
    str
        The output_vcf filename supplied in Inputs.
    """
    out_dir_name = os.path.dirname(output_vcf)

    # add filter status to info fields
    logger.debug("add filter status to info fields")
    out_vcf_with_info_filter = f"{output_vcf}.tmp"
    cmd = [
        "bcftools",
        "annotate",
        "--threads",
        "30",
        "-c",
        "INFO/filter_status:=FILTER",
        "-O",
        "z",
        "-o",
        out_vcf_with_info_filter,
        input_vcf,
    ]
    logger.debug(cmd)
    subprocess.check_call(cmd)
    info_field_tags["filter_status"] = ["filter status", "String"]

    # remove old annotation files if they exist
    annotation_file = pjoin(out_dir_name, "annot.txt.gz")
    if os.path.exists(annotation_file):
        logger.debug(f"Removing existing annotation file: {annotation_file}")
        os.remove(annotation_file)
    annotation_index_file = pjoin(out_dir_name, "annot.txt.gz.tbi")
    if os.path.exists(annotation_index_file):
        logger.debug(f"Removing existing annotation index file: {annotation_index_file}")
        os.remove(annotation_index_file)
    hdr_file = pjoin(out_dir_name, "hdr.txt")
    if os.path.exists(hdr_file):
        logger.debug(f"Removing existing header file: {hdr_file}")
        os.remove(hdr_file)

    # create query string and columns string
    logger.debug("create query string and columns string ")
    query_string = ""
    c_string = ""
    for info_tag in info_field_tags:
        query_string = f"{query_string}\t%{info_tag}"
        c_string = f"{c_string},FORMAT/{info_tag}"
    for info_tag in lucos_info_fields:
        query_string = f"{query_string}\t%{info_tag}"
        c_string = f"{c_string},INFO/{info_tag}"

    # Extract all INFO/{tag} into a tab-delimited annotation file
    logger.debug(r"Extract all INFO/\{tag\} into a tab-delimited annotation file")

    cmd = [
        "bcftools",
        "query",
        "-f",
        f"%CHROM\t%POS\t%REF\t%ALT{query_string}\\n",
        out_vcf_with_info_filter,  # <-- your input VCF here
    ]

    with open(annotation_file, "wb") as annot_out:
        # 1. Launch bcftools query, capturing its stdout
        p1 = subprocess.Popen(cmd, stdout=subprocess.PIPE)

        # 2. Launch bgzip, reading from p1.stdout and writing to your file
        p2 = subprocess.Popen(["bgzip", "-c"], stdin=p1.stdout, stdout=annot_out)  # noqa: S607

        # 3. Close the write end in the parent so that bgzip sees EOF
        p1.stdout.close()

        # 4. Wait for both to finish
        p1.wait()
        p2.wait()

    logger.debug(" ".join(cmd))

    # Index the file with tabix
    logger.debug("Index the file with tabix")
    cmd = ["tabix", "-s1", "-b2", "-e2", annotation_file]
    logger.debug(" ".join(cmd))
    subprocess.check_call(cmd)

    # Create a header lines for the new annotation tags
    logger.debug("Create a header lines for the new annotation tags")
    for info_tag in info_field_tags:
        header_line = (
            f"##FORMAT=<ID={info_tag},Number=1,Type={info_field_tags[info_tag][1]},"
            f"Description={info_field_tags[info_tag][0]}>"
        )
        with open(hdr_file, "a") as hdr_out:
            hdr_out.write(header_line + "\n")
    for info_tag in lucos_info_fields:
        header_line = (
            f"##INFO=<ID={info_tag},Number=1,Type={lucos_info_fields[info_tag][1]},"
            f"Description={lucos_info_fields[info_tag][0]}>"
        )
        with open(hdr_file, "a") as hdr_out:
            hdr_out.write(header_line + "\n")

    # Transfer the annotation to sample 'SAMPLE'
    logger.debug("Transfer the annotation to sample SAMPLE")
    max_threads = os.cpu_count()
    cmd = [
        "bcftools",
        "annotate",
        "--threads",
        str(max_threads),
        "-x",
        "INFO",
        "-s",
        "SAMPLE",
        "-a",
        annotation_file,
        "-h",
        hdr_file,
        "-c",
        f"CHROM,POS,REF,ALT{c_string}",
        "-O",
        "z",
        "-o",
        output_vcf,
        out_vcf_with_info_filter,
    ]
    logger.debug(cmd)
    subprocess.check_call(cmd)
    # Index the output VCF with tabix
    subprocess.check_call(["tabix", "-p", "vcf", output_vcf])  # noqa: S607


def merge_vcf_files(tumor_vcf_info_to_format, normal_vcf_info_to_format, out_merged_vcf):
    """
    Merge tumor and normal VCF files into a single VCF file.

    Parameters
    ----------
    tumor_vcf_info_to_format : str
        Path to the tumor VCF file with INFO fields moved to FORMAT.
    normal_vcf_info_to_format : str
        Path to the normal VCF file with INFO fields moved to FORMAT.
    out_merged_vcf : str
        Path to the output merged VCF file.

    Returns
    -------
    str
        Path to the output merged VCF file with tumor-PASS variants only.
    """
    max_threads = os.cpu_count()

    # merging T-N VCF files - this results with records from both tumor and normal VCF files
    cmd_merge = [
        "bcftools",
        "merge",
        "--threads",
        str(max_threads),
        "-m",
        "none",
        "--force-samples",
        "-Oz",
        "-o",
        out_merged_vcf,
        tumor_vcf_info_to_format,
        normal_vcf_info_to_format,
    ]
    logger.debug(" ".join(cmd_merge))
    subprocess.check_call(cmd_merge)
    cmd_index = ["bcftools", "index", "-t", out_merged_vcf]
    logger.debug(" ".join(cmd_index))
    subprocess.check_call(cmd_index)

    # Filtering for tumor-PASS variants only
    cmd_filter = [
        "bcftools",
        "view",
        "-i",
        'FORMAT/filter_status[0] == "PASS"',
        out_merged_vcf,
        "-Oz",
        "-o",
        out_merged_vcf.replace(".vcf.gz", ".tumor_PASS.vcf.gz"),
    ]
    logger.debug(" ".join(cmd_filter))
    subprocess.check_call(cmd_filter)
    # Index the filtered VCF
    cmd_index = ["bcftools", "index", "-t", out_merged_vcf.replace(".vcf.gz", ".tumor_PASS.vcf.gz")]
    logger.debug(" ".join(cmd_index))
    subprocess.check_call(cmd_index)

    return out_merged_vcf.replace(".vcf.gz", ".tumor_PASS.vcf.gz")


def integrate_tandem_repeat_features(merged_vcf, ref_tr_file, out_dir):
    # Use a temporary directory for all intermediate files
    with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
        # generate tandem repeat info
        df_merged_vcf = vcftools.get_vcf_df(merged_vcf)
        df_merged_vcf.insert(
            2, "end", df_merged_vcf["pos"] + 1
        )  # TBD: get the actual end coordinate when the variant is not SNV (Insertion).
        bed1 = pjoin(tmpdir, "merged_vcf.tmp.bed")
        df_merged_vcf[["chrom", "pos", "end"]].to_csv(bed1, sep="\t", header=None, index=False)
        # sort the reference tandem repeat file
        ref_tr_file_sorted = pjoin(tmpdir, "ref_tr_file.sorted.bed")
        cmd = ["bedtools", "sort", "-i", ref_tr_file]
        with open(ref_tr_file_sorted, "w") as sorted_file:
            subprocess.check_call(cmd, stdout=sorted_file)
        # find closest tandem-repeat for each variant
        bed2 = ref_tr_file_sorted
        bed1_with_closest_tr_tsv = pjoin(tmpdir, "merged_vcf.tmp.TRdata.tsv")
        cmd_bedtools = ["bedtools", "closest", "-D", "ref", "-a", bed1, "-b", bed2]
        cmd_cut = ["cut", "-f1-2,5-8"]
        with open(bed1_with_closest_tr_tsv, "w") as out_file:
            p1 = subprocess.Popen(cmd_bedtools, stdout=subprocess.PIPE)
            p2 = subprocess.Popen(cmd_cut, stdin=p1.stdout, stdout=out_file)
            p1.stdout.close()
            p2.communicate()
            p1.wait()

        df_tr_info = pd.read_csv(bed1_with_closest_tr_tsv, sep="\t", header=None)
        df_tr_info.columns = ["chrom", "pos", "TR_start", "TR_end", "TR_seq", "TR_distance"]
        df_tr_info["TR_length"] = df_tr_info["TR_end"] - df_tr_info["TR_start"]
        # Extract repeat unit length, handle cases where pattern does not match
        extracted = df_tr_info["TR_seq"].str.extract(r"\((\w+)\)")
        df_tr_info["TR_seq_unit_length"] = extracted[0].str.len()
        # Fill NaN values with 0 and log a warning if any were found
        if df_tr_info["TR_seq_unit_length"].isna().any():
            logger.warning(
                "Some TR_seq values did not match the expected pattern '(unit)'. "
                "Setting TR_seq_unit_length to 0 for these rows."
            )
            df_tr_info["TR_seq_unit_length"] = df_tr_info["TR_seq_unit_length"].fillna(0).astype(int)
        df_tr_info.to_csv(bed1_with_closest_tr_tsv, sep="\t", header=None, index=False)

        sorted_tsv = pjoin(tmpdir, "merged_vcf.tmp.TRdata.sorted.tsv")
        cmd = ["sort", "-k1,1", "-k2,2n", bed1_with_closest_tr_tsv]
        with open(sorted_tsv, "w") as out_file:
            subprocess.check_call(cmd, stdout=out_file)
        gz_tsv = sorted_tsv + ".gz"
        cmd = ["bgzip", "-c", sorted_tsv]
        with open(gz_tsv, "wb") as out_file:
            subprocess.check_call(cmd, stdout=out_file)
        cmd = ["tabix", "-s", "1", "-b", "2", "-e", "2", gz_tsv]
        subprocess.check_call(cmd)

        # integrate tandem repeat info into the merged vcf file
        hdr_txt = [
            '##INFO=<ID=TR_start,Number=1,Type=String,Description="Closest tandem Repeat Start">',
            '##INFO=<ID=TR_end,Number=1,Type=String,Description="Closest Tandem Repeat End">',
            '##INFO=<ID=TR_seq,Number=1,Type=String,Description="Closest Tandem Repeat Sequence">',
            '##INFO=<ID=TR_distance,Number=1,Type=String,Description="Closest Tandem Repeat Distance">',
            '##INFO=<ID=TR_length,Number=1,Type=String,Description="Closest Tandem Repeat total length">',
            '##INFO=<ID=TR_seq_unit_length,Number=1,Type=String,Description="Closest Tandem Repeat unit length">',
        ]
        hdr_file = pjoin(tmpdir, "tr_hdr.txt")
        with open(hdr_file, "w") as f:
            f.writelines(line + "\n" for line in hdr_txt)
        merged_vcf_with_tr_info = merged_vcf.replace(".vcf.gz", ".tr_info.vcf.gz")
        cmd = [
            "bcftools",
            "annotate",
            "-Oz",
            "-o",
            merged_vcf_with_tr_info,
            "-a",
            gz_tsv,
            "-h",
            hdr_file,
            "-c",
            "CHROM,POS,INFO/TR_start,INFO/TR_end,INFO/TR_seq,INFO/TR_distance,INFO/TR_length,INFO/TR_seq_unit_length",
            merged_vcf,
        ]
        subprocess.check_call(cmd)
    return merged_vcf_with_tr_info


def __parse_args(argv: list[str]) -> argparse.Namespace:
    """
    Parse command line arguments.

    Parameters
    ----------
    argv : list of str
        Command line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="create_somatic_pileup_featuremap.py",
        description=run.__doc__,
    )
    parser.add_argument("--tumor_vcf", help="tumor vcf file", required=True, type=str)
    parser.add_argument("--normal_vcf", help="normal vcf file", required=True, type=str)
    parser.add_argument("--sample_name", help="sample_name", required=True, type=str)
    parser.add_argument("--ref_tr_file", help="reference tandem repeat file", required=True, type=str)
    parser.add_argument(
        "--out_directory",
        help="out directory where intermediate and output files will be saved."
        " if not supplied all files will be written to current directory",
        required=False,
        type=str,
        default=".",
    )
    parser.add_argument(
        "--filter_for_tumor_pass_variants",
        help="If set, the output VCF will only contain tumor-PASS variants.",
        action="store_true",
        default=False,
    )
    return parser.parse_args(argv[1:])


def run(argv):
    """
    Merge two VCF files (tumor and normal) into a single VCF file.

    The output VCF file will have all tumor records merged with corresponding normal records,
    with INFO fields moved to FORMAT.

    Parameters
    ----------
    argv : list of str
        Command line arguments.

    Returns
    -------
    None
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

    # Set up the output VCF file path
    out_merged_vcf = pjoin(args.out_directory, f"{args.sample_name}.tumor_normal.merged.vcf.gz")
    logger.info(f"Output merged VCF file: {out_merged_vcf}")

    # Get the features used in the featuremap_xgb_prediction module
    info_field_tags, lucos_info_fields = get_combined_vcf_features()

    # Move INFO fields to FORMAT in the tumor and normal VCF files
    tumor_vcf_info_to_format = pjoin(args.out_directory, f"{args.sample_name}.tumor.info_to_format.vcf.gz")
    normal_vcf_info_to_format = pjoin(args.out_directory, f"{args.sample_name}.normal.info_to_format.vcf.gz")
    move_vcf_value_from_INFO_to_FORMAT(args.tumor_vcf, tumor_vcf_info_to_format, info_field_tags, lucos_info_fields)
    move_vcf_value_from_INFO_to_FORMAT(args.normal_vcf, normal_vcf_info_to_format, info_field_tags, lucos_info_fields)
    # Merge the tumor and normal VCF files into a single VCF file
    if args.filter_for_tumor_pass_variants:
        logger.info("Filtering for tumor-PASS variants only.")
        out_merged_vcf_tumor_pass = merge_vcf_files(tumor_vcf_info_to_format, normal_vcf_info_to_format, out_merged_vcf)
        out_merged_vcf_tumor_pass = integrate_tandem_repeat_features(
            out_merged_vcf_tumor_pass, args.ref_tr_file, args.out_directory
        )
        logger.info(f"Merged VCF file created: {out_merged_vcf}")
        logger.info(f"Merged VCF tumor-PASS file created: {out_merged_vcf_tumor_pass}")
    else:
        logger.info("No filtering for tumor-PASS variants. Merging all records from both VCF files.")


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
