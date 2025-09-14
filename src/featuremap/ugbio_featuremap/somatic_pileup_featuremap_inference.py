#!/env/python
# Copyright 2023 Ultima Genomics Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# DESCRIPTION
#    Run somatic pileup featuremap inference using an input somatic featuremap pileup VCF file
#    and a pre-trained XGBoost model.
# CHANGELOG in reverse chronological order

import argparse
import logging
import os
import re
import subprocess
import sys
import tempfile
from os.path import basename, dirname
from os.path import join as pjoin

import numpy as np
import pandas as pd
import pysam
import xgboost
from ugbio_core.logger import logger
from ugbio_core.vcf_pipeline_utils import VcfPipelineUtils
from ugbio_core.vcfbed import vcftools

from ugbio_featuremap import featuremap_xgb_prediction

TR_CUSTOM_INFO_FIELDS = ["TR_distance", "TR_length", "TR_seq_unit_length"]
vpu = VcfPipelineUtils()


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
        prog="somatic_pileup_featuremap_inference.py",
        description=(
            "Run somatic pileup featuremap inference using an input somatic featuremap pileup "
            "VCF file and a pre-trained XGBoost model."
        ),
    )
    parser.add_argument(
        "--in_sfmp",
        help="Input somatic featuremap pileup file.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--xgb_model",
        help="Path to the trained XGBoost model file.",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--out_directory",
        help="Output directory where results will be saved. Defaults to current directory.",
        required=False,
        type=str,
        default=".",
    )
    return parser.parse_args(argv[1:])


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

    # merge dataframes
    df_tumor_normal = pd.concat([df_tumor.add_prefix("t_"), df_normal.add_prefix("n_")], axis=1)

    # create merged fillna dict
    if fillna_dict:
        fillna_dict_merged = {}
        for key in fillna_dict:  # noqa: PLC0206
            fillna_dict_merged[f"t_{key}"] = fillna_dict[key]
            fillna_dict_merged[f"n_{key}"] = fillna_dict[key]
        df_tumor_normal = df_tumor_normal.fillna(fillna_dict_merged)

    # TR info fields are stored as model features in their original naming.
    for tr_field in TR_CUSTOM_INFO_FIELDS:
        key = f"t_{tr_field.lower()}"
        df_tumor_normal[tr_field] = df_tumor_normal[key]

    return df_tumor_normal


def load_model(xgb_model_file: str) -> "xgboost.XGBClassifier":
    """
    Load a pre-trained XGBoost model from a file.

    Args:
        xgb_model_file (str): Path to the XGBoost model file.

    Returns:
        xgboost.XGBClassifier: The loaded XGBoost classifier model.
    """
    # load xgb model
    xgb_clf_es = xgboost.XGBClassifier()
    xgb_clf_es.load_model(xgb_model_file)
    return xgb_clf_es


def predict(xgb_model: "xgboost.XGBClassifier", df_calls: "pd.DataFrame") -> "np.ndarray":
    """
    Generate prediction probabilities for the positive class using a trained XGBoost model.

    Args:
        xgb_model (xgboost.XGBClassifier): Trained XGBoost classifier with accessible feature names.
        df_calls (pd.DataFrame): DataFrame containing feature columns required by the model.

    Returns:
        np.ndarray: Array of predicted probabilities for the positive class ("1").
    """
    model_features = xgb_model.get_booster().feature_names
    X = df_calls[model_features]  # noqa: N806

    featuremap_xgb_prediction.set_categorial_columns(X)
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category")

    probabilities = xgb_model.predict_proba(X)
    df_probabilities = pd.DataFrame(probabilities, columns=["0", "1"])
    return df_probabilities["1"].to_numpy()


def map_fields_to_vcf(vcf_path, lowercase_fields):
    """
    Given a VCF file and a list of lowercase field names,
    return the actual field names in the VCF header (INFO and FORMAT).

    Parameters
    ----------
    vcf_path : str
        Path to VCF file (bgzipped .vcf.gz).
    lowercase_fields : list[str]
        List of field names (lowercase).

    Returns
    -------
    dict
        Mapping from lowercase field name -> actual VCF field name (if found).

    # Example output: {'dp': 'DP', 'af': 'AF', 'ref_m2': 'REF_M2'}
    """
    vcf = pysam.VariantFile(vcf_path)

    # Extract all INFO and FORMAT fields
    info_fields = set(vcf.header.info.keys())
    format_fields = set(vcf.header.formats.keys())
    all_fields = info_fields | format_fields

    # Build case-insensitive lookup
    lookup = {field.lower(): field for field in all_fields}
    # Also add exact matches to handle fields that are already correctly cased
    lookup.update({field: field for field in all_fields})

    # Map requested lowercase names to actual names (if they exist)
    result = {}
    custom_info_fields = []
    for lf in lowercase_fields:
        lf_sub = re.sub(r"^[tn]_", "", lf)
        result[lf_sub] = lookup.get(lf_sub, None)  # None if not found
        custom_info_fields.append(result[lf_sub])

    return result, custom_info_fields


def annotate_xgb_proba_to_vcf(df_sfmp: pd.DataFrame, in_sfmp_vcf: str, out_vcf: str) -> None:
    """
    Annotate the VCF file with XGBoost probabilities.

    Parameters
    ----------
    df_sfmp : pd.DataFrame
        DataFrame containing the XGBoost probabilities.
    in_sfmp_vcf : str
        Path to the input VCF file.
    out_vcf : str
        Path to the output annotated VCF file.

    Returns
    -------
    None
    """
    # Prepare the DataFrame for annotation
    tsv_df = df_sfmp[["t_chrom", "t_pos", "t_ref", "t_alt_allele", "xgb_proba"]].copy()
    tsv_df.columns = ["#CHROM", "POS", "REF", "ALT", "xgb_proba"]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as tmp_tsv:
        tsv_path = tmp_tsv.name
        tsv_df.to_csv(tmp_tsv, sep="\t", index=False, float_format="%.15f")
        cmd = ["sort", "-k1,1", "-k2,2n", tsv_path]
        sorted_tsv = pjoin(dirname(tsv_path), "sorted_" + basename(tsv_path))
        with open(sorted_tsv, "w") as out_file:
            subprocess.check_call(cmd, stdout=out_file)
        gz_tsv = sorted_tsv + ".gz"
        cmd = ["bgzip", "-c", sorted_tsv]
        with open(gz_tsv, "wb") as out_file:
            subprocess.check_call(cmd, stdout=out_file)
        cmd = ["tabix", "-s", "1", "-b", "2", "-e", "2", gz_tsv]
        subprocess.check_call(cmd)

    # Create a temporary header file for the new INFO field
    header_line = '##INFO=<ID=xgb_proba,Number=1,Type=Float,Description="XGBoost probability">\n'
    with tempfile.NamedTemporaryFile(mode="w", suffix=".hdr", delete=False) as tmp_hdr:
        hdr_path = tmp_hdr.name
        tmp_hdr.write(header_line)

    # Annotate using bcftools
    cmd = [
        "bcftools",
        "annotate",
        "-a",
        gz_tsv,
        "-c",
        "CHROM,POS,REF,ALT,xgb_proba",
        "-h",
        hdr_path,
        "-o",
        out_vcf,
        "-O",
        "z",
        in_sfmp_vcf,
    ]
    logger.debug(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    # Index the output VCF
    vpu.index_vcf(out_vcf)


def run(argv):
    """
    This function performs the following steps:
    - Parses command-line arguments.
    - Sets up logging.
    - Loads an XGBoost model.
    - Maps model-features to VCF-fields.
    - Reads and processes a merged tumor-normal VCF file to a dataframe.
    - Predicts probabilities using the loaded model.
    - Saves the results as a Parquet file.
    - Annotates the original VCF with the predicted probabilities.

    Parameters
    ----------
    argv : list
        List of command-line arguments.

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

    # Load model
    xgb_clf_es = load_model(args.xgb_model)
    model_features = xgb_clf_es.get_booster().feature_names
    logger.info(f"loaded model. model features: {model_features}")
    __fields_mapping, custom_info_fields = map_fields_to_vcf(args.in_sfmp, model_features)
    logger.debug(f"model features fields mapping: {__fields_mapping}")

    # Read somatic-featuremap-pileup-vcf into dataframe
    df_sfmp = read_merged_tumor_normal_vcf(args.in_sfmp, custom_info_fields=custom_info_fields)

    # Inference
    df_sfmp["xgb_proba"] = predict(xgb_clf_es, df_sfmp)

    #  Write outputs
    df_sfmp.to_parquet(pjoin(args.out_directory, "df_sfmp.new.parquet"))
    logger.debug(f"dataframe with xgb_proba is saved in: {pjoin(args.out_directory,'df_sfmp.new.parquet')}")

    out_sfmp_with_xgb_proba = os.path.basename(args.in_sfmp).replace("vcf.gz", "xgb_proba.vcf.gz")
    annotate_xgb_proba_to_vcf(df_sfmp, args.in_sfmp, pjoin(args.out_directory, out_sfmp_with_xgb_proba))
    logger.info(f"Annotated VCF with XGBoost probabilities saved to: {out_sfmp_with_xgb_proba}")


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
