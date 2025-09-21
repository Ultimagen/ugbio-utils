import argparse
import logging
import os
import statistics
import subprocess
import sys
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor
from os.path import basename, dirname
from os.path import join as pjoin

import pandas as pd
import pysam
from ugbio_core.logger import logger
from ugbio_core.vcfbed import vcftools

from ugbio_featuremap import somatic_pileup_featuremap_inference
from ugbio_featuremap.featuremap_utils import FeatureMapFields

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

############################################################
############################################################
# columns_for_SRV_training (not used in SRSNV model v1.23)
# GT	1	String	Genotype
# DP_FILT	1	Integer	Number of reads containing this location that pass the adjacent base filter
# RAW_VAF	1	Float	Raw VAF := N_alt_reads/N_total_reads
# VAF	1	Float	VAF := N_alt_reads/(N_ref_reads+N_alt_reads)
# AD	A	Integer	Number of reads supporting the reference allele in locus
# AD_A	1	Integer	Number of reads supporting the base A in locus
# AD_C	1	Integer	Number of reads supporting the base C in locus
# AD_G	1	Integer	Number of reads supporting the base G in locus
# AD_T	1	Integer	Number of reads supporting the base T in locus
# AD_DEL	1	Integer	Number of reads supporting a deletion in locus
# AD_INS	1	Integer	Number of reads supporting an adjacent insertion in locus
# DUP	.	Integer	Is the read a duplicate, interpreted from CRAM flag
# MAPQ	.	Integer	Read mapping quality
# MQUAL	.	Float	SingleReadSNV model inferred raw Phred scaled quality
# SNVQ	.	Float	SingleReadSNV model inferred Phred scaled quality, recalibrated to the SNVQ value
# FILT	.	Integer	Pre-filter status for SNVs reads,
#           1 means an SNVs read passed all the filters defined in the SRSNV model,
#           0 means it failed at least one filter
# DP_MAPQ60	1	Integer	Number of reads with mapping of at least 60 containing this locus
# ADJ_REF_DIFF	.	Integer	The 3 adjacent bases to the locus do not fully match the reference genome
############################################################
############################################################

# TBD : to make it more robust , maybe we can do the following:
# given a vcf,get all info/format fields
# go over the records - if fields is integer/float mark the field name for aggregation


info_fields_for_training = ["TR_distance", "TR_length", "TR_seq_unit_length"]
format_fields_for_training = ["DP_FILT", "RAW_VAF", "VAF", "DP_MAPQ60", "ADJ_REF_DIFF", "ref_count", "non_ref_count"]
added_format_features = {
    "alt_reads": ["number of supporting reads for the alternative allele", "Integer"],
    "pass_alt_reads": ["number of passed supporting reads for the alternative allele", "Integer"],
    "mqual_mean": ["mean value of MQUAL", "Float"],
    "snvq_mean": ["mean value of SNVQ", "Float"],
    "mqual_max": ["mean value of MQUAL", "Float"],
    "snvq_max": ["mean value of SNVQ", "Float"],
    "mqual_min": ["mean value of MQUAL", "Float"],
    "snvq_min": ["mean value of SNVQ", "Float"],
    "count_duplicate": ["number of duplicate reads", "Integer"],
    "count_non_duplicate": ["number of non-duplicate reads", "Integer"],
}
added_info_features = {
    "ref_allele": ["reference allele", "String"],
    "alt_allele": ["alternative allele", "String"],
}
columns_for_aggregation = [FeatureMapFields.MQUAL.value, FeatureMapFields.SNVQ.value]


def process_sample_columns(df_variants, prefix):
    def aggregate_mean(df, colname):
        values = []
        for tup in df[colname]:
            cleaned_list = list(tup)
            values.append(statistics.mean(cleaned_list))
        return values

    def aggregate_min(df, colname):
        values = []
        for tup in df[colname]:
            cleaned_list = list(tup)
            values.append(min(cleaned_list))
        return values

    def aggregate_max(df, colname):
        values = []
        for tup in df[colname]:
            cleaned_list = list(tup)
            values.append(max(cleaned_list))
        return values

    def parse_is_duplicate(df, dup_colname):
        df["count_duplicate"] = df[dup_colname].apply(lambda x: sum(x))
        df["count_non_duplicate"] = df[dup_colname].apply(lambda x: sum(1 for val in x if val == 0))
        return df

    def parse_padding_ref_counts(df_variants, ref_counts_colname, non_ref_counts_colname):
        # Handle ref_count
        padding_counts_length = len(df_variants[ref_counts_colname].iloc[0])
        ref_df = pd.DataFrame(
            df_variants[ref_counts_colname].tolist(), columns=[f"{prefix}ref{i}" for i in range(padding_counts_length)]
        )
        df_variants = pd.concat([df_variants, ref_df], axis=1)
        # Handle nonref_count
        nonref_df = pd.DataFrame(
            df_variants[non_ref_counts_colname].tolist(),
            columns=[f"{prefix}nonref{i}" for i in range(padding_counts_length)],
        )
        df_variants = pd.concat([df_variants, nonref_df], axis=1)
        return df_variants

    """Process columns for a sample with given prefix (t_ or n_)"""
    # Process alt_reads
    df_variants[f"{prefix}alt_reads"] = [tup[1] for tup in df_variants[f"{prefix}ad"]]
    # Process pass_alt_reads
    df_variants[f"{prefix}pass_alt_reads"] = df_variants[f"{prefix}{FeatureMapFields.FILT.value.lower()}"].apply(
        lambda x: sum(x)
    )
    # Process aggregations for each column
    for colname in columns_for_aggregation:
        colname_lower = colname.lower()
        df_variants[f"{prefix}{colname_lower}_mean"] = aggregate_mean(df_variants, f"{prefix}{colname_lower}")
        df_variants[f"{prefix}{colname_lower}_max"] = aggregate_max(df_variants, f"{prefix}{colname_lower}")
        df_variants[f"{prefix}{colname_lower}_min"] = aggregate_min(df_variants, f"{prefix}{colname_lower}")

    # Process duplicates
    df_variants = parse_is_duplicate(df_variants, f"{prefix}{FeatureMapFields.DUP.value.lower()}")
    df_variants = parse_padding_ref_counts(df_variants, f"{prefix}ref_counts", f"{prefix}nonref_counts")

    return df_variants


def df_sfm_fields_transformation(df_variants):  # noqa: C901
    # Process both tumor and normal samples
    for prefix in ["t_", "n_"]:
        df_variants = process_sample_columns(df_variants, prefix)
    df_variants["ref_allele"] = [tup[0] for tup in df_variants["alleles"]]
    df_variants["alt_allele"] = [tup[1] for tup in df_variants["alleles"]]

    return df_variants


def sort_and_filter_vcf(featuremap_vcf_file, temp_dir, filter_string, interval_srting):
    sorted_featuremap = pjoin(temp_dir, basename(featuremap_vcf_file).replace(".vcf.gz", ".sorted.vcf.gz"))
    sorted_filtered_featuremap = pjoin(
        temp_dir, basename(featuremap_vcf_file).replace(".vcf.gz", ".sorted.filtered.vcf.gz")
    )
    sort_cmd = f"bcftools view {featuremap_vcf_file} {interval_srting} |\
                bcftools sort - -Oz -o {sorted_featuremap} && \
                bcftools index -t {sorted_featuremap}"
    logger.debug(sort_cmd)
    subprocess.check_call(sort_cmd, shell=True)  # noqa: S602
    if filter_string != "":
        sort_cmd = f"bcftools view {filter_string} {featuremap_vcf_file} {interval_srting} |\
                    bcftools sort - -Oz -o {sorted_filtered_featuremap} && \
                    bcftools index -t {sorted_filtered_featuremap}"
        logger.debug(sort_cmd)
        subprocess.check_call(sort_cmd, shell=True)  # noqa: S602
    else:
        sorted_filtered_featuremap = sorted_featuremap
    return sorted_featuremap, sorted_filtered_featuremap


def add_fields_to_header(hdr, added_format_features, added_info_features):
    for field in added_format_features:
        field_type = added_format_features[field][1]
        field_description = added_format_features[field][0]
        hdr.formats.add(field, 1, field_type, field_description)
    for field in added_info_features:
        field_type = added_info_features[field][1]
        field_description = added_info_features[field][0]
        hdr.info.add(field, 1, field_type, field_description)


def process_vcf_row(row, df_variants, hdr, vcfout, write_agg_params):
    pos = row.pos
    chrom = row.chrom
    alt_allele = row.alleles[1]
    df_record = df_variants[
        (df_variants["chrom"] == chrom) & (df_variants["pos"] == pos) & (df_variants["alt_allele"] == alt_allele)
    ]

    if len(df_record) > 0:
        if write_agg_params:
            for key in added_info_features:
                row.info[key] = df_record[key].to_list()[0]
            for key in added_format_features:
                row.samples[0][key] = df_record[f"t_{key}"].to_list()[0]
                row.samples[1][key] = df_record[f"n_{key}"].to_list()[0]
            if "xgb_proba" in hdr.info:
                row.info["xgb_proba"] = df_record["xgb_proba"].to_list()[0]

    vcfout.write(row)


def read_merged_tumor_normal_vcf(
    vcf_file: str, custom_info_fields: list[str], fillna_dict: dict[str, object] = None, chrom: str = None
) -> "pd.DataFrame":
    """
    Reads a merged tumor-normal VCF file and returns a concatenated DataFrame with prefixed columns
    for tumor and normal samples.
    Args:
        vcf_file (str): Path to the VCF file containing both tumor and normal samples.
        custom_info_fields (list[str]): List of custom INFO fields to extract from the VCF.
        fillna_dict (dict[str, object], optional): Dictionary specifying values to fill missing data
            for each field. Defaults to None.
        chrom (str, optional): Chromosome to filter the VCF records. If None, all chromosomes are
            included. Defaults to None.
    Returns:
        pd.DataFrame: A DataFrame with tumor columns prefixed by 't_' and normal columns
            prefixed by 'n_'. Missing values are filled according to `fillna_dict` if provided.
    """
    # Read to df
    if chrom is not None:
        df_tumor = vcftools.get_vcf_df(vcf_file, sample_id=0, custom_info_fields=custom_info_fields, chromosome=chrom)
        df_normal = vcftools.get_vcf_df(vcf_file, sample_id=1, custom_info_fields=custom_info_fields, chromosome=chrom)
    else:
        df_tumor = vcftools.get_vcf_df(vcf_file, sample_id=0, custom_info_fields=custom_info_fields)
        df_normal = vcftools.get_vcf_df(vcf_file, sample_id=1, custom_info_fields=custom_info_fields)

    # make all colnames in lowercase.
    for colname in df_tumor.columns:
        df_tumor[colname.lower()] = df_tumor[colname]
        df_normal[colname.lower()] = df_normal[colname]
        if colname != colname.lower():
            df_tumor = df_tumor.drop(columns=[colname])
            df_normal = df_normal.drop(columns=[colname])

    # merge dataframes
    df_tumor_normal = pd.concat([df_tumor.add_prefix("t_"), df_normal.add_prefix("n_")], axis=1)

    # create merged fillna dict
    if fillna_dict:
        fillna_dict_merged = {}
        for key in fillna_dict:  # noqa: PLC0206
            fillna_dict_merged[f"t_{key}"] = fillna_dict[key]
            fillna_dict_merged[f"n_{key}"] = fillna_dict[key]
        df_tumor_normal = df_tumor_normal.fillna(fillna_dict_merged)

    return df_tumor_normal


def featuremap_fields_aggregation(  # noqa: C901
    somatic_featuremap_vcf_file: str,
    output_vcf: str,
    filter_tags=None,
    genomic_interval: str = None,
    xgb_model_file: str = None,
    write_agg_params: bool = True,  # noqa: FBT001, FBT002
    verbose: bool = True,  # noqa: FBT001, FBT002
) -> str:
    """
    Write the vcf file with the aggregated fields and the xgb probability
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    # filter vcf file for the given filter tags and genomic interval
    filter_string = f"-f {filter_tags}" if filter_tags else ""
    interval_srting = genomic_interval if genomic_interval else ""
    with tempfile.TemporaryDirectory(dir=dirname(output_vcf)) as temp_dir:
        sorted_featuremap, sorted_filtered_featuremap = sort_and_filter_vcf(
            somatic_featuremap_vcf_file, temp_dir, filter_string, interval_srting
        )

        # get custom-info-fields and read vcf block to dataframe
        custom_info_fields = info_fields_for_training + format_fields_for_training + columns_for_aggregation
        df_variants = read_merged_tumor_normal_vcf(sorted_filtered_featuremap, custom_info_fields=custom_info_fields)

        df_variants = df_sfm_fields_transformation(df_variants)
        if xgb_model_file:
            xgb_clf_es = somatic_pileup_featuremap_inference.load_model(xgb_model_file)
            model_features = xgb_clf_es.get_booster().feature_names
            logger.info(f"loaded model. model features: {model_features}")
            df_variants["xgb_proba"] = somatic_pileup_featuremap_inference.predict(xgb_clf_es, df_variants)

        with pysam.VariantFile(sorted_featuremap) as vcfin:
            hdr = vcfin.header
            add_fields_to_header(hdr)
            if xgb_model_file:
                hdr.info.add("xgb_proba", 1, "Float", "XGBoost model predicted probability")
            with pysam.VariantFile(output_vcf, mode="w", header=hdr) as vcfout:
                for row in vcfin:
                    process_vcf_row(row, df_variants, hdr, vcfout, write_agg_params)
            vcfout.close()
            vcfin.close()
    pysam.tabix_index(output_vcf, preset="vcf", min_shift=0, force=True)
    return output_vcf


def featuremap_fields_aggregation_on_an_interval_list(
    featuremap_vcf_file: str,
    output_vcf: str,
    interval_list: str,
    filter_tags=None,
    verbose: bool = True,  # noqa: FBT001, FBT002
) -> None:
    """
    Apply featuremap fields aggregation on an interval list
    Inputs:
        featuremap (str): The input featuremap vcf file
        output_vcf (str): The output pileup vcf file
        interval_list (str): The interval list file
        verbose (bool): The verbosity level (default: True)
    Output:
        output_vcf (str): The output vcf file including the aggregated fields and the xgb probability
    """
    if not output_vcf.endswith(".vcf.gz"):
        logger.debug("adding .vcf.gz suffix to the output vcf file")
        output_vcf = output_vcf + ".vcf.gz"

    with tempfile.TemporaryDirectory(dir=dirname(output_vcf)):
        genomic_intervals = []
        with open(interval_list, encoding="utf-8") as f:
            for line in f:
                # ignore header lines
                if line.startswith("@"):
                    continue
                # read genomic ineterval
                genomic_interval = line.strip()
                genomic_interval_list = genomic_interval.split("\t")
                chrom = genomic_interval_list[0]
                start = genomic_interval_list[1]
                end = genomic_interval_list[2]
                genomic_interval = chrom + ":" + str(start) + "-" + str(end)
                genomic_intervals.append(genomic_interval)

        params = [
            (
                featuremap_vcf_file,
                f"{output_vcf}.{genomic_interval}.int_list.vcf.gz",
                filter_tags,
                genomic_interval,
                verbose,
            )
            for genomic_interval in genomic_intervals
        ]
        num_cpus = os.cpu_count()
        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            results = list(executor.map(lambda p: featuremap_fields_aggregation(*p), params))

        # Write each string to the file
        with open("interval_vcf_files.list", "w") as file:
            for interval_vcf_file in results:
                file.write(interval_vcf_file + "\n")

        cmd = (
            f"bcftools concat -f interval_vcf_files.list -a | "
            f"bcftools sort - -Oz -o {output_vcf} && "
            f"bcftools index -t {output_vcf}"
        )
        logger.debug(cmd)
        subprocess.check_call(cmd, shell=True)  # noqa: S602
    return output_vcf


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="somatic featuremap fields transformation",
        description=run.__doc__,
    )
    parser.add_argument(
        "-f",
        "--somatic featuremap",
        type=str,
        required=True,
        help="""somatic featuremap vcf file""",
    )
    parser.add_argument(
        "-filter_string",
        "--filter_string",
        type=str,
        required=False,
        default="PASS",
        help="""filter tags to apply on the somatic featuremap pileup vcf file""",
    )
    parser.add_argument(
        "-o",
        "--output_vcf",
        type=str,
        required=True,
        help="""Output pileup vcf file""",
    )
    parser.add_argument(
        "-i",
        "--interval_list_file",
        type=str,
        required=True,
        help="""Interval list file""",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        type=bool,
        required=False,
        default=True,
        help="""Whether to print debug messages (default: True)""",
    )
    return parser.parse_args(argv[1:])


def run(argv):
    """Add aggregated parameters and xgb probability to the featuremap pileup vcf file"""
    args_in = __parse_args(argv)
    featuremap_fields_aggregation_on_an_interval_list(
        args_in.somatic_featuremap,
        args_in.output_vcf,
        args_in.interval_list_file,
        args_in.filter_string,
        args_in.verbose,
    )


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
