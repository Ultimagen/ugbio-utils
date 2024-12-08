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
from ugbio_ppmseq.ppmSeq_consts import HistogramColumnNames

from ugbio_featuremap import featuremap_consensus_utils
from ugbio_featuremap.featuremap_utils import FeatureMapFields

warnings.simplefilter(action="ignore", category=FutureWarning)


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
columns_for_min_aggregation = [FeatureMapFields.X_QUAL.value, FeatureMapFields.X_INDEX.value]
columns_for_max_aggregation = [FeatureMapFields.X_QUAL.value, FeatureMapFields.X_INDEX.value]
columns_for_fillna = [FeatureMapFields.IS_CYCLE_SKIP.value]
columns_for_st_et_aggregation = ["st", "et"]

added_agg_features = {
    "alt_reads": ["number of supporting reads for the alternative allele", "Integer"],
    "ref_allele": ["reference allele", "String"],
    "alt_allele": ["alternative allele", "String"],
    "X_QUAL_mean": ["mean value of X_QUAL", "Float"],
    "X_SCORE_mean": ["mean value of X_SCORE", "Float"],
    "X_EDIST_mean": ["mean value of X_EDIST", "Float"],
    "X_LENGTH_mean": ["mean value of X_LENGTH", "Float"],
    "X_MAPQ_mean": ["mean value of X_MAPQ", "Float"],
    "X_FC1_mean": ["mean value of X_FC1", "Float"],
    "X_FC2_mean": ["mean value of X_FC2", "Float"],
    "MAX_SOFTCLIP_LENGTH_mean": ["mean value of MAX_SOFTCLIP_LENGTH", "Float"],
    "X_FLAGS_mean": ["mean value of X_FLAGS", "Float"],
    "ML_QUAL_mean": ["mean value of ML_QUAL", "Float"],
    "X_QUAL_max": ["max value of X_QUAL", "Float"],
    "X_INDEX_max": ["max value of X_INDEX", "Integer"],
    "X_QUAL_min": ["min value of X_QUAL", "Float"],
    "X_INDEX_min": ["min value of X_INDEX", "Integer"],
    "count_forward": ["number of forward reads", "Integer"],
    "count_reverse": ["number of reverse reads", "Integer"],
    "count_duplicate": ["number of duplicate reads", "Integer"],
    "count_non_duplicate": ["number of non-duplicate reads", "Integer"],
}

ppm_added_agg_features = {}
ppm_added_agg_features["st_MINUS"] = ["number of st tagged as MINUS", "Integer"]
ppm_added_agg_features["st_MIXED"] = ["number of st tagged as MIXED", "Integer"]
ppm_added_agg_features["st_PLUS"] = ["number of st tagged as PLUS", "Integer"]
ppm_added_agg_features["st_UNDETERMINED"] = ["number of st tagged as UNDETERMINED", "Integer"]
ppm_added_agg_features["et_MINUS"] = ["number of et tagged as MINUS", "Integer"]
ppm_added_agg_features["et_MIXED"] = ["number of et tagged as MIXED", "Integer"]
ppm_added_agg_features["et_PLUS"] = ["number of et tagged as PLUS", "Integer"]
ppm_added_agg_features["et_UNDETERMINED"] = ["number of et tagged as UNDETERMINED", "Integer"]
ppm_added_agg_features["num_mixed_reads"] = ["number of mixed reads", "Integer"]


def record_manual_aggregation(rec, xgb_model=None):  # noqa: C901
    record_info_dict = dict(rec.info)
    record_dict_for_xgb = {}

    # add non aggrgate fields
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
        agg_colname = colname.upper() + "_mean"
        record_dict_for_xgb[agg_colname] = statistics.mean(record_info_dict[colname])
    for colname in columns_for_min_aggregation:
        agg_colname = colname.upper() + "_min"
        record_dict_for_xgb[agg_colname] = min(record_info_dict[colname])
    for colname in columns_for_max_aggregation:
        agg_colname = colname.upper() + "_max"
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
                record_dict_for_xgb[colname + "_" + tag] = record_info_dict[colname].split("|").count(tag)

    if xgb_model:
        record_dict_for_xgb["xgb_proba"] = predict_record_with_xgb(record_dict_for_xgb, xgb_model)

    return record_dict_for_xgb


def set_categorial_columns(df):
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    # print(categorical_columns)
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
            "count_forward": "num_is_forward",
            "VAF": "vaf",
        }
    )

    X = xgb_df[features]  # noqa: N806
    set_categorial_columns(X)
    X = X.fillna(0)  # noqa: N806

    # predict record
    probabilities = xgb_clf_es.predict_proba(X)
    df_probabilities = pd.DataFrame(probabilities, columns=["0", "1"])
    return df_probabilities["1"].to_numpy()[0]


def pileup_featuremap_with_agg_params_and_xgb_proba(  # noqa: C901
    featuremap_vcf_file: str,
    output_vcf: str,
    genomic_interval: str = None,
    xgb_model: xgboost.XGBClassifier = None,
    verbose: bool = True,  # noqa: FBT001, FBT002
) -> str:
    """
    Write the vcf file with the aggregated fields and the xgb probability
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    with tempfile.TemporaryDirectory(dir=dirname(output_vcf)) as temp_dir:
        sorted_featuremap = pjoin(temp_dir, basename(featuremap_vcf_file).replace(".vcf.gz", ".sorted.vcf.gz"))
        if genomic_interval is None:
            # sort all featuremap
            sort_cmd = f"bcftools sort {featuremap_vcf_file} -Oz -o {sorted_featuremap} && bcftools index -t {sorted_featuremap}"  # noqa: E501
        else:
            # sort only the genomic interval of interest
            sort_cmd = f"bcftools view {featuremap_vcf_file} {genomic_interval} |\
                    bcftools sort - -Oz -o {sorted_featuremap} && bcftools index -t {sorted_featuremap}"
        logger.debug(sort_cmd)
        subprocess.check_call(sort_cmd, shell=True)  # noqa: S602

        with pysam.VariantFile(sorted_featuremap) as vcfin:
            hdr = vcfin.header
            # adding manual aggregation fields
            # for field, field_type, field_description in zip(
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
            with pysam.VariantFile(output_vcf, mode="w", header=hdr) as vcfout:
                for row in vcfin:
                    record_dict_for_xgb = record_manual_aggregation(row, xgb_model)
                    for key in added_agg_features:
                        row.info[key] = record_dict_for_xgb[key]
                    if "st" in hdr.info:
                        for key in ppm_added_agg_features:
                            row.info[key] = record_dict_for_xgb[key]
                    row.info["xgb_proba"] = float(record_dict_for_xgb["xgb_proba"])
                    vcfout.write(row)
            vcfout.close()
            vcfin.close()
    pysam.tabix_index(output_vcf, preset="vcf", min_shift=0, force=True)
    return output_vcf


def pileup_featuremap_with_agg_params_and_xgb_proba_on_an_interval_list(
    featuremap_vcf_file: str,
    output_vcf: str,
    interval_list: str,
    xgb_model: str = None,
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
                genomic_interval,
                xgb_model,
                verbose,
            )
            for genomic_interval in genomic_intervals
        ]
        num_cpus = os.cpu_count()
        with ThreadPoolExecutor(max_workers=num_cpus) as executor:
            results = list(executor.map(lambda p: pileup_featuremap_with_agg_params_and_xgb_proba(*p), params))

        # merge the output vcfs
        vcf_str = " ".join(results)
        cmd = f"bcftools concat {vcf_str} -a | bcftools sort - -Oz -o {output_vcf} && bcftools index -t {output_vcf}"
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
        args_in.xgb_model_file,
        args_in.verbose,
    )


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
