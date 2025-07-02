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
import xgboost
from sklearn.preprocessing import LabelEncoder
from ugbio_core.logger import logger
from ugbio_core.vcfbed import vcftools
from ugbio_ppmseq.ppmSeq_consts import HistogramColumnNames

from ugbio_featuremap import featuremap_consensus_utils
from ugbio_featuremap.featuremap_utils import FeatureMapFields

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


default_custom_info_fields = [
    FeatureMapFields.X_SCORE.value,
    FeatureMapFields.X_EDIST.value,
    FeatureMapFields.X_LENGTH.value,
    FeatureMapFields.X_MAPQ.value,
    FeatureMapFields.X_INDEX.value,
    FeatureMapFields.X_FC1.value,
    FeatureMapFields.X_FC2.value,
    FeatureMapFields.MAX_SOFTCLIP_LENGTH.value,
    FeatureMapFields.X_FLAGS.value,
    HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value,
    HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value,
    "ML_QUAL",
    FeatureMapFields.X_RN.value,
    FeatureMapFields.X_CIGAR.value,
    "rq",
    "tm",
    FeatureMapFields.IS_FORWARD.value,
    FeatureMapFields.IS_DUPLICATE.value,
    FeatureMapFields.READ_COUNT.value,
    FeatureMapFields.FILTERED_COUNT.value,
    FeatureMapFields.TRINUC_CONTEXT_WITH_ALT.value,
    FeatureMapFields.HMER_CONTEXT_REF.value,
    FeatureMapFields.HMER_CONTEXT_ALT.value,
    FeatureMapFields.PREV_1.value,
    FeatureMapFields.PREV_2.value,
    FeatureMapFields.PREV_3.value,
    FeatureMapFields.NEXT_1.value,
    FeatureMapFields.NEXT_2.value,
    FeatureMapFields.NEXT_3.value,
    FeatureMapFields.IS_CYCLE_SKIP.value,
    FeatureMapFields.X_QUAL.value,
]

ppm_custom_info_fields = ["st", "et"]

columns_for_mean_aggregation = [
    FeatureMapFields.X_QUAL.value,
    FeatureMapFields.X_SCORE.value,
    FeatureMapFields.X_EDIST.value,
    FeatureMapFields.X_LENGTH.value,
    FeatureMapFields.X_MAPQ.value,
    FeatureMapFields.X_FC1.value,
    FeatureMapFields.X_FC2.value,
    FeatureMapFields.MAX_SOFTCLIP_LENGTH.value,
    FeatureMapFields.X_FLAGS.value,
    "ML_QUAL",
]
columns_for_min_aggregation = [FeatureMapFields.X_QUAL.value.lower(), FeatureMapFields.X_INDEX.value.lower()]
columns_for_max_aggregation = [FeatureMapFields.X_QUAL.value.lower(), FeatureMapFields.X_INDEX.value.lower()]
columns_for_fillna = [FeatureMapFields.IS_CYCLE_SKIP.value]
columns_for_st_et_aggregation = ["st", "et"]

added_agg_features = {
    "alt_reads": ["number of supporting reads for the alternative allele", "Integer"],
    "ref_allele": ["reference allele", "String"],
    "alt_allele": ["alternative allele", "String"],
    "x_qual_mean": ["mean value of X_QUAL", "Float"],
    "x_score_mean": ["mean value of X_SCORE", "Float"],
    "x_edist_mean": ["mean value of X_EDIST", "Float"],
    "x_length_mean": ["mean value of X_LENGTH", "Float"],
    "x_mapq_mean": ["mean value of X_MAPQ", "Float"],
    "x_fc1_mean": ["mean value of X_FC1-Number of M bases different on read from references", "Float"],
    "x_fc2_mean": ["mean value of X_FC2-Number of features before score threshold filter", "Float"],
    "max_softclip_length_mean": ["mean value of MAX_SOFTCLIP_LENGTH", "Float"],
    "x_flags_mean": ["mean value of X_FLAGS", "Float"],
    "ml_qual_mean": ["mean value of ML_QUAL", "Float"],
    "x_qual_max": ["max value of X_QUAL", "Float"],
    "x_index_max": ["max value of X_INDEX", "Integer"],
    "x_qual_min": ["min value of X_QUAL", "Float"],
    "x_index_min": ["min value of X_INDEX", "Integer"],
    "count_forward": ["number of forward reads", "Integer"],
    "count_reverse": ["number of reverse reads", "Integer"],
    "count_duplicate": ["number of duplicate reads", "Integer"],
    "count_non_duplicate": ["number of non-duplicate reads", "Integer"],
}

ppm_added_agg_features = {
    "st_minus": ["number of st tagged as MINUS", "Integer"],
    "st_mixed": ["number of st tagged as MIXED", "Integer"],
    "st_plus": ["number of st tagged as PLUS", "Integer"],
    "st_undetermined": ["number of st tagged as UNDETERMINED", "Integer"],
    "et_minus": ["number of et tagged as MINUS", "Integer"],
    "et_mixed": ["number of et tagged as MIXED", "Integer"],
    "et_plus": ["number of et tagged as PLUS", "Integer"],
    "et_undetermined": ["number of et tagged as UNDETERMINED", "Integer"],
    "num_mixed_reads": ["number of mixed reads", "Integer"],
}


def record_manual_aggregation(rec, xgb_model=None):  # noqa: C901
    record_info_dict = dict(rec.info)
    record_dict_for_xgb = {}
    # add non aggrgate fields
    record_dict_for_xgb["POS"] = rec.pos
    record_dict_for_xgb["alt_reads"] = rec.samples[0]["AD"][1]
    record_dict_for_xgb["ref_allele"] = rec.alleles[0]
    record_dict_for_xgb["alt_allele"] = rec.alleles[1]
    record_dict_for_xgb["ref"] = rec.ref
    record_dict_for_xgb["qual"] = rec.qual
    non_agg_fields_for_xgb_from_rec = ["DP", "VAF"]
    for field in non_agg_fields_for_xgb_from_rec:
        record_dict_for_xgb[field] = rec.samples[0][field]
    non_agg_fields_for_xgb_from_rec_dict = featuremap_consensus_utils.fields_to_collect_all_options[
        "fields_to_write_once"
    ]
    for field in non_agg_fields_for_xgb_from_rec_dict:
        record_dict_for_xgb[field] = record_info_dict[field]

    # add aggregate fields
    for colname in columns_for_mean_aggregation:
        agg_colname = colname.lower() + "_mean"
        record_dict_for_xgb[agg_colname] = statistics.mean(record_info_dict[colname])
    for colname in columns_for_min_aggregation:
        agg_colname = colname.lower() + "_min"
        record_dict_for_xgb[agg_colname] = min(record_info_dict[colname])
    for colname in columns_for_max_aggregation:
        agg_colname = colname.lower() + "_max"
        record_dict_for_xgb[agg_colname] = max(record_info_dict[colname])
    record_dict_for_xgb["count_forward"] = record_info_dict["is_forward"].split("|").count("T")
    record_dict_for_xgb["count_reverse"] = record_info_dict["is_forward"].split("|").count("F")
    record_dict_for_xgb["count_duplicate"] = record_info_dict["is_duplicate"].split("|").count(True)
    record_dict_for_xgb["count_non_duplicate"] = record_info_dict["is_duplicate"].split("|").count(False)
    if "is_cycle_skip" in record_info_dict:
        record_dict_for_xgb["is_cycle_skip"] = record_info_dict["is_cycle_skip"]
    else:
        record_dict_for_xgb["is_cycle_skip"] = False

    if "st" in record_info_dict:  # ppmseq columns
        record_dict_for_xgb["num_mixed_reads"] = sum(
            [
                1
                for i, j in zip(record_info_dict["st"], record_info_dict["et"], strict=False)
                if (i == j) & (i == "MIXED")
            ]
        )
        for colname in columns_for_st_et_aggregation:
            tags = ["MINUS", "MIXED", "PLUS", "UNDETERMINED"]
            for tag in tags:
                record_dict_for_xgb[colname + "_" + tag.lower()] = record_info_dict[colname].split("|").count(tag)

    # print(record_dict_for_xgb)
    if xgb_model:
        record_dict_for_xgb["xgb_proba"] = predict_record_with_xgb(record_dict_for_xgb, xgb_model)

    return record_dict_for_xgb


def df_vcf_manual_aggregation(df_variants, xgb_model=None):  # noqa: C901
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

    def parse_st_et_fields(df, colname):
        tags = ["MINUS", "MIXED", "PLUS", "UNDETERMINED"]
        for tag in tags:
            df[colname + "_" + tag.lower()] = df[colname].apply(lambda x, tag=tag: x.split("|").count(tag))
        return df

    def calculate_num_mixed(df):
        num_mixed = []
        for _index, row in df.iterrows():
            st, et = row["st"], row["et"]
            st_list = st.split("|")
            et_list = et.split("|")
            sum_value = sum(1 for i, j in zip(st_list, et_list, strict=False) if (i == j) & (i == "MIXED"))
            num_mixed.append(sum_value)
        df["num_mixed_reads"] = num_mixed
        return df

    def parse_is_forward(df):
        df["count_forward"] = df["is_forward"].apply(lambda x: x.split("|").count("T"))
        df["count_reverse"] = df["is_forward"].apply(lambda x: x.split("|").count("F"))
        return df

    def parse_is_duplicate(df):
        df["count_duplicate"] = df["is_duplicate"].apply(lambda x: x.split("|").count("T"))
        df["count_non_duplicate"] = df["is_duplicate"].apply(lambda x: x.split("|").count("F"))
        return df

    df_variants["alt_reads"] = [tup[1] for tup in df_variants["ad"]]
    df_variants["ref_allele"] = [tup[0] for tup in df_variants["alleles"]]
    df_variants["alt_allele"] = [tup[1] for tup in df_variants["alleles"]]
    for colname in columns_for_mean_aggregation:
        agg_colname = colname.lower() + "_mean"
        df_variants[agg_colname] = aggregate_mean(df_variants, colname.lower())
    for colname in columns_for_max_aggregation:
        agg_colname = colname.lower() + "_max"
        df_variants[agg_colname] = aggregate_max(df_variants, colname.lower())
    for colname in columns_for_min_aggregation:
        agg_colname = colname.lower() + "_min"
        df_variants[agg_colname] = aggregate_min(df_variants, colname.lower())
    df_variants = parse_is_forward(df_variants)
    df_variants = parse_is_duplicate(df_variants)
    df_variants["is_cycle_skip"] = df_variants["is_cycle_skip"].fillna(value=False)

    if "st" in df_variants.columns:
        for colname in columns_for_st_et_aggregation:
            df_variants = parse_st_et_fields(df_variants, colname.lower())
        df_variants = calculate_num_mixed(df_variants)

    if (xgb_model is not None) & (len(df_variants) > 0):
        df_variants["xgb_proba"] = predict_record_with_xgb(df_variants, xgb_model)

    return df_variants


def set_categorial_columns(df):
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    le = LabelEncoder()
    for col in categorical_columns:
        df.loc[:, col] = le.fit_transform(df[col].astype(str))


def predict_record_with_xgb(record_dict_for_xgb, xgb_model):
    """
    Predict the record with the xgb model
    Inputs:
        record_dict_for_xgb (dict): A dictionary containing the record information
        xgb_model (xgboost.XGBClassifier): The xgboost model
    Output:
        probability_value (float): The xgb probability of the record to be a true variant
    """

    # load xgb model
    xgb_clf_es = xgboost.XGBClassifier()
    xgb_clf_es.load_model(xgb_model)
    features = xgb_clf_es.get_booster().feature_names
    # prepare dataframe
    xgb_df = pd.DataFrame(record_dict_for_xgb, index=[0])
    xgb_df = xgb_df.rename(
        columns={
            "DP": "dp",
            "X_READ_COUNT": "x_read_count",
            "X_FILTERED_COUNT": "x_filtered_count",
            "VAF": "vaf",
            "POS": "pos",
        }
    )

    X = xgb_df[features]  # noqa: N806
    set_categorial_columns(X)
    X = X.fillna(0)  # noqa: N806

    # predict record
    probabilities = xgb_clf_es.predict_proba(X)
    df_probabilities = pd.DataFrame(probabilities, columns=["0", "1"])
    return df_probabilities["1"].to_numpy()


def predict_record_with_xgb(df_variants, xgb_model):  # noqa: F811
    """
    Predict the record with the xgb model
    Inputs:
        variants dataframe (pd.DataFrame): A dataframe containing variants information
        xgb_model (xgboost.XGBClassifier): The xgboost model
    Output:
        probability_value (float): The xgb probability of the record to be a true variant
    """

    # load xgb model
    xgb_clf_es = xgboost.XGBClassifier()
    xgb_clf_es.load_model(xgb_model)
    features = xgb_clf_es.get_booster().feature_names

    X = df_variants[features]  # noqa: N806
    set_categorial_columns(X)
    X = X.fillna(0).infer_objects(copy=False)  # noqa: N806

    # predict record
    probabilities = xgb_clf_es.predict_proba(X)
    df_probabilities = pd.DataFrame(probabilities, columns=["0", "1"])
    return df_probabilities["1"].to_numpy()


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


def add_agg_fields_to_header(hdr):
    for field in added_agg_features:
        field_type = added_agg_features[field][1]
        field_description = added_agg_features[field][0]
        hdr.info.add(field, 1, field_type, field_description)
    if "st" in hdr.info:
        for field in ppm_added_agg_features:
            field_type = ppm_added_agg_features[field][1]
            field_description = ppm_added_agg_features[field][0]
            hdr.info.add(field, 1, field_type, field_description)
    hdr.info.add("xgb_proba", 1, "Float", "XGBoost probability of the record to be a true variant")


def process_vcf_row(row, df_variants, hdr, vcfout, write_agg_params):
    pos = row.pos
    chrom = row.chrom
    alt_allele = row.alleles[1]
    df_record = df_variants[
        (df_variants["chrom"] == chrom) & (df_variants["pos"] == pos) & (df_variants["alt_allele"] == alt_allele)
    ]

    if len(df_record) > 0:
        if write_agg_params:
            for key in added_agg_features:
                row.info[key] = df_record[key].to_list()[0]
            if "st" in hdr.info:
                for key in ppm_added_agg_features:
                    row.info[key] = df_record[key].to_list()[0]
        if "xgb_proba" in df_record.columns:
            row.info["xgb_proba"] = float(df_record["xgb_proba"].to_list()[0])
    vcfout.write(row)


def pileup_featuremap_with_agg_params_and_xgb_proba(  # noqa: C901
    featuremap_vcf_file: str,
    output_vcf: str,
    filter_tags=None,
    genomic_interval: str = None,
    xgb_model: xgboost.XGBClassifier = None,
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

    filter_string = f"-f {filter_tags}" if filter_tags else ""
    interval_srting = genomic_interval if genomic_interval else ""
    with tempfile.TemporaryDirectory(dir=dirname(output_vcf)) as temp_dir:
        sorted_featuremap, sorted_filtered_featuremap = sort_and_filter_vcf(
            featuremap_vcf_file, temp_dir, filter_string, interval_srting
        )

        # read vcf block to dataframe
        custom_info_fields = default_custom_info_fields
        custom_info_fields.extend(ppm_custom_info_fields)
        df_variants = vcftools.get_vcf_df(sorted_filtered_featuremap, custom_info_fields=custom_info_fields)
        df_variants = df_vcf_manual_aggregation(df_variants, xgb_model)

        with pysam.VariantFile(sorted_featuremap) as vcfin:
            hdr = vcfin.header
            add_agg_fields_to_header(hdr)
            with pysam.VariantFile(output_vcf, mode="w", header=hdr) as vcfout:
                for row in vcfin:
                    process_vcf_row(row, df_variants, hdr, vcfout, write_agg_params)
            vcfout.close()
            vcfin.close()
    pysam.tabix_index(output_vcf, preset="vcf", min_shift=0, force=True)
    return output_vcf


def pileup_featuremap_with_agg_params_and_xgb_proba_on_an_interval_list(
    featuremap_vcf_file: str,
    output_vcf: str,
    interval_list: str,
    filter_tags=None,
    xgb_model: str = None,
    write_agg_params: bool = True,  # noqa: FBT001, FBT002
    verbose: bool = True,  # noqa: FBT001, FBT002
) -> None:
    """
    Apply pileup featuremap on an interval list
    Inputs:
        featuremap (str): The input featuremap vcf file
        output_vcf (str): The output pileup vcf file
        interval_list (str): The interval list file
        xgb_model (str): The xgboost model file
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
                xgb_model,
                write_agg_params,
                verbose,
            )
            for genomic_interval in genomic_intervals
        ]
        num_cpus = os.cpu_count()
        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            results = list(executor.map(lambda p: pileup_featuremap_with_agg_params_and_xgb_proba(*p), params))

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
        prog="add_aggregate_parameters_and_xgb_score_to_pileup_featuremap",
        description=run.__doc__,
    )
    parser.add_argument(
        "-f",
        "--featuremap_pileup",
        type=str,
        required=True,
        help="""Featuremap pileup vcf file""",
    )
    parser.add_argument(
        "-filter_string",
        "--filter_string",
        type=str,
        required=False,
        default="PASS",
        help="""filter tags to apply on the featuremap pileup vcf file""",
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
        "-m",
        "--xgb_model_file",
        type=str,
        required=True,
        help="""XGBoost model file""",
    )
    parser.add_argument(
        "-write_agg_params",
        "--write_agg_params",
        type=bool,
        required=False,
        default=True,
        help="""Whether to write the aggregated parameters (default: True)""",
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
    pileup_featuremap_with_agg_params_and_xgb_proba_on_an_interval_list(
        args_in.featuremap_pileup,
        args_in.output_vcf,
        args_in.interval_list_file,
        args_in.filter_string,
        args_in.xgb_model_file,
        args_in.write_agg_params,
        args_in.verbose,
    )


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
