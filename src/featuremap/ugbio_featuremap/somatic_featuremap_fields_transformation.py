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

info_fields_for_training = ["TR_DISTANCE", "TR_LENGTH", "TR_SEQ_UNIT_LENGTH"]

format_fields_for_training = [
    FeatureMapFields.VAF.value,
    FeatureMapFields.RAW_VAF.value,
    FeatureMapFields.FILT.value,
    FeatureMapFields.DP_FILT.value,
    FeatureMapFields.ADJ_REF_DIFF.value,
    FeatureMapFields.DP_MAPQ60.value,
    FeatureMapFields.DUP.value,
]
format_mpileup_fields_for_training = ["ref_counts_pm_2", "nonref_counts_pm_2"]

added_format_features = {
    "ALT_READS": ["number of supporting reads for the alternative allele", "Integer"],
    "PASS_ALT_READS": ["number of passed supporting reads for the alternative allele", "Integer"],
    "MQUAL_MEAN": ["mean value of MQUAL", "Float"],
    "SNVQ_MEAN": ["mean value of SNVQ", "Float"],
    "MQUAL_MAX": ["mean value of MQUAL", "Float"],
    "SNVQ_MAX": ["mean value of SNVQ", "Float"],
    "MQUAL_MIN": ["mean value of MQUAL", "Float"],
    "SNVQ_MIN": ["mean value of SNVQ", "Float"],
    "COUNT_DUPLICATE": ["number of duplicate reads", "Integer"],
    "COUNT_NON_DUPLICATE": ["number of non-duplicate reads", "Integer"],
}
added_info_features = {
    "REF_ALLELE": ["reference allele", "String"],
    "ALT_ALLELE": ["alternative allele", "String"],
}
columns_for_aggregation = [FeatureMapFields.MQUAL.value, FeatureMapFields.SNVQ.value]


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
        df_tr_info.columns = ["chrom", "pos", "TR_START", "TR_END", "TR_SEQ", "TR_DISTANCE"]
        df_tr_info["TR_LENGTH"] = df_tr_info["TR_END"] - df_tr_info["TR_START"]
        # Extract repeat unit length, handle cases where pattern does not match
        extracted = df_tr_info["TR_SEQ"].str.extract(r"\((\w+)\)")
        df_tr_info["TR_SEQ_UNIT_LENGTH"] = extracted[0].str.len()
        # Fill NaN values with 0 and log a warning if any were found
        if df_tr_info["TR_SEQ_UNIT_LENGTH"].isna().any():
            logger.warning(
                "Some TR_SEQ values did not match the expected pattern '(unit)'. "
                "Setting TR_SEQ_UNIT_LENGTH to 0 for these rows."
            )
            df_tr_info["TR_SEQ_UNIT_LENGTH"] = df_tr_info["TR_SEQ_UNIT_LENGTH"].fillna(0).astype(int)
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
            '##INFO=<ID=TR_START,Number=1,Type=String,Description="Closest tandem Repeat Start">',
            '##INFO=<ID=TR_END,Number=1,Type=String,Description="Closest Tandem Repeat End">',
            '##INFO=<ID=TR_SEQ,Number=1,Type=String,Description="Closest Tandem Repeat Sequence">',
            '##INFO=<ID=TR_DISTANCE,Number=1,Type=String,Description="Closest Tandem Repeat Distance">',
            '##INFO=<ID=TR_LENGTH,Number=1,Type=String,Description="Closest Tandem Repeat total length">',
            '##INFO=<ID=TR_SEQ_UNIT_LENGTH,Number=1,Type=String,Description="Closest Tandem Repeat unit length">',
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
            "CHROM,POS,INFO/TR_START,INFO/TR_END,INFO/TR_SEQ,INFO/TR_DISTANCE,INFO/TR_LENGTH,INFO/TR_SEQ_UNIT_LENGTH",
            merged_vcf,
        ]
        subprocess.check_call(cmd)

    pysam.tabix_index(merged_vcf_with_tr_info, preset="vcf", min_shift=0, force=True)
    return merged_vcf_with_tr_info


def process_sample_columns(df_variants, prefix):  # noqa: C901
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

    def parse_is_duplicate(df, dup_colname, prefix):
        df[f"{prefix}count_duplicate"] = df[dup_colname].apply(
            lambda x: sum(x) if x is not None and len(x) > 0 and None not in x else float("nan")
        )
        df[f"{prefix}count_non_duplicate"] = df[dup_colname].apply(
            lambda x: sum(1 for val in x if val == 0)
            if x is not None and len(x) > 0 and None not in x
            else float("nan")
        )
        return df

    def parse_padding_ref_counts(df_variants, ref_counts_colname, non_ref_counts_colname):
        # Handle ref_count
        padding_counts_length = len(df_variants[ref_counts_colname].iloc[0])
        ref_df = pd.DataFrame(
            df_variants[ref_counts_colname].tolist(), columns=[f"{prefix}ref{i}" for i in range(padding_counts_length)]
        )
        for col in ref_df.columns:
            df_variants[col] = ref_df[col]
        # Handle nonref_count
        nonref_df = pd.DataFrame(
            df_variants[non_ref_counts_colname].tolist(),
            columns=[f"{prefix}nonref{i}" for i in range(padding_counts_length)],
        )
        for col in nonref_df.columns:
            df_variants[col] = nonref_df[col]

        return df_variants

    """Process columns for a sample with given prefix (t_ or n_)"""
    # Process alt_reads
    df_variants[f"{prefix}alt_reads"] = [tup[1] for tup in df_variants[f"{prefix}ad"]]
    # Process pass_alt_reads;
    df_variants[f"{prefix}pass_alt_reads"] = df_variants[f"{prefix}{FeatureMapFields.FILT.value.lower()}"].apply(
        lambda x: sum(x) if x is not None and len(x) > 0 and None not in x else float("nan")
    )
    # Process aggregations for each column
    for colname in columns_for_aggregation:
        colname_lower = colname.lower()
        df_variants[f"{prefix}{colname_lower}_mean"] = aggregate_mean(df_variants, f"{prefix}{colname_lower}")
        df_variants[f"{prefix}{colname_lower}_max"] = aggregate_max(df_variants, f"{prefix}{colname_lower}")
        df_variants[f"{prefix}{colname_lower}_min"] = aggregate_min(df_variants, f"{prefix}{colname_lower}")

    # Process duplicates
    df_variants = parse_is_duplicate(df_variants, f"{prefix}{FeatureMapFields.DUP.value.lower()}", prefix)
    df_variants = parse_padding_ref_counts(
        df_variants,
        f"{prefix}{format_mpileup_fields_for_training[0]}",
        f"{prefix}{format_mpileup_fields_for_training[1]}",
    )

    return df_variants


def df_sfm_fields_transformation(df_variants):  # noqa: C901
    # Process both tumor and normal samples
    for prefix in ["t_", "n_"]:
        df_variants = process_sample_columns(df_variants, prefix)
    df_variants["ref_allele"] = [tup[0] for tup in df_variants["t_alleles"]]
    df_variants["alt_allele"] = [tup[1] for tup in df_variants["t_alleles"]]

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
        (df_variants["t_chrom"] == chrom) & (df_variants["t_pos"] == pos) & (df_variants["alt_allele"] == alt_allele)
    ]

    if len(df_record) > 0:
        if write_agg_params:
            for key in added_info_features:
                row.info[key.upper()] = df_record[key.lower()].to_list()[0]
            for key in added_format_features:
                tumor_value = df_record[f"t_{key.lower()}"].to_list()[0]
                normal_value = df_record[f"n_{key.lower()}"].to_list()[0]
                if pd.notna(tumor_value):
                    row.samples[0][key.upper()] = tumor_value
                else:
                    row.samples[0][key.upper()] = None
                if pd.notna(normal_value):
                    row.samples[1][key.upper()] = normal_value
                else:
                    row.samples[1][key.upper()] = None
            if "XGB_PROBA" in hdr.info:
                row.info["XGB_PROBA"] = df_record["xgb_proba"].to_list()[0]

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

        custom_info_fields = (
            format_fields_for_training
            + format_mpileup_fields_for_training
            + info_fields_for_training
            + columns_for_aggregation
        )
        df_variants = read_merged_tumor_normal_vcf(sorted_filtered_featuremap, custom_info_fields=custom_info_fields)
        df_variants = df_sfm_fields_transformation(df_variants)

        if xgb_model_file is not None:
            xgb_clf_es = somatic_pileup_featuremap_inference.load_model(xgb_model_file)
            model_features = xgb_clf_es.get_booster().feature_names
            logger.info(f"loaded model. model features: {model_features}")
            df_variants["xgb_proba"] = somatic_pileup_featuremap_inference.predict(xgb_clf_es, df_variants)
        # Write df_variants to parquet file
        parquet_output = output_vcf.replace(".vcf.gz", "_featuremap.parquet")
        df_variants.to_parquet(parquet_output, index=False)
        logger.info(f"Written feature map dataframe to {parquet_output}")

        # TBD: vcf writing takes long time - guess its beacauee of the lookup for each record
        with pysam.VariantFile(sorted_featuremap) as vcfin:
            hdr = vcfin.header
            add_fields_to_header(hdr, added_format_features, added_info_features)
            if xgb_model_file is not None:
                hdr.info.add("XGB_PROBA", 1, "Float", "XGBoost model predicted probability")
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
    xgb_model_file: str = None,
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
                xgb_model_file,
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
        "-sfm",
        "--somatic_featuremap",
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
        "-ref_tr",
        "--ref_tr_file",
        type=str,
        required=True,
        help="""Reference tandem repeat file in BED format""",
    )
    parser.add_argument(
        "-xgb_model",
        "--xgb_model_file",
        type=str,
        required=False,
        default=None,
        help="""XGBoost model file for inference (optional)""",
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

    # add tandem repeat features
    out_dir = dirname(args_in.output_vcf)
    sfmp_with_tr = integrate_tandem_repeat_features(args_in.somatic_featuremap, args_in.ref_tr_file, out_dir)

    featuremap_fields_aggregation_on_an_interval_list(
        sfmp_with_tr,
        args_in.output_vcf,
        args_in.interval_list_file,
        args_in.filter_string,
        args_in.xgb_model_file,
        args_in.verbose,
    )


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
