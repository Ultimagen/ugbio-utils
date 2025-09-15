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
import sys
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


def parse_padding_ref_counts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the padding ref counts from the DataFrame and create new columns for each padding position.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the 't_ref_counts_pm_*' and 'n_ref_counts_pm_*' columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with new columns for each padding position added.
    """
    # Identify the padding ref count columns
    # Find columns matching the padding ref count pattern
    ref_count_pattern = re.compile(r"^(ref|nonref)_counts_pm_(\d+)$")
    padding_columns = [col for col in df.columns if ref_count_pattern.match(col)]

    if not padding_columns:
        raise ValueError("No padding ref count columns found in DataFrame")

    # Extract the distance from center (should be the same for all columns)
    match = ref_count_pattern.match(padding_columns[0])
    distance_from_center = int(match.group(2))

    # Verify all columns have the same padding distance
    for col in padding_columns:
        match = ref_count_pattern.match(col)
        if match and int(match.group(2)) != distance_from_center:
            raise ValueError(f"Inconsistent padding distances found: {distance_from_center} vs {match.group(2)}")
    # Create new column-name for each padding position
    ref_colnames = []
    nonref_colnames = []
    for loc in np.arange(-distance_from_center, distance_from_center + 1, 1):
        if loc < 0:
            ref_colnames.append(f"ref_m{abs(loc)}")
            nonref_colnames.append(f"nonref_m{abs(loc)}")
        else:
            ref_colnames.append(f"ref_{loc}")
            nonref_colnames.append(f"nonref_{loc}")

    # Split the lists in the original columns into separate columns
    ref_colname = f"ref_counts_pm_{distance_from_center}"
    split_df = pd.DataFrame(df[ref_colname].tolist(), columns=ref_colnames)
    for new_col in split_df.columns:
        df[new_col] = split_df[new_col].values  # noqa: PD011

    nonref_colname = f"nonref_counts_pm_{distance_from_center}"
    split_df = pd.DataFrame(df[nonref_colname].tolist(), columns=nonref_colnames)
    for new_col in split_df.columns:
        df[new_col] = split_df[new_col].values  # noqa: PD011

    return df


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

    # parse padding ref counts into separate columns
    df_tumor = parse_padding_ref_counts(df_tumor)
    df_normal = parse_padding_ref_counts(df_normal)

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

    ### Add padding ref counts fields if a distance was found
    # find padding value
    distance_from_center = None
    for field in format_fields:
        match = re.match(r"^(ref|nonref)_counts_pm_(\d+)$", field)
        if match:
            distance_from_center = match.group(2)
            break
    # Add padding ref counts fields if a distance was found
    if distance_from_center is not None:
        for prefix in ("ref", "nonref"):
            field_name = f"{prefix}_counts_pm_{distance_from_center}"
            if field_name not in custom_info_fields:
                custom_info_fields.append(field_name)

    return result, custom_info_fields


def annotate_xgb_proba_to_vcf(df_sfmp: pd.DataFrame, in_sfmp_vcf: str, out_vcf: str) -> None:
    """
    Annotate the VCF file with XGBoost probabilities using pysam only, memory-efficiently.

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
    # Serially walk both VCF and DataFrame, adding INFO only when variant matches
    # Build an iterator over the DataFrame, sorted by chrom, pos, ref, alt
    df_sorted = df_sfmp.sort_values(["t_chrom", "t_pos", "t_ref", "t_alt_allele"]).reset_index(drop=True)
    df_iter = df_sorted.itertuples(index=False)
    try:
        df_row = next(df_iter)
    except StopIteration:
        df_row = None

    with pysam.VariantFile(in_sfmp_vcf) as invcf:
        new_header = invcf.header.copy()
        if "xgb_proba" not in new_header.info:
            new_header.add_line('##INFO=<ID=xgb_proba,Number=1,Type=Float,Description="XGBoost probability">')
        with pysam.VariantFile(out_vcf, "wz", header=new_header) as outvcf:
            for rec in invcf:
                # Advance df_row until it matches or passes the VCF record
                while df_row is not None:
                    new_record = VcfPipelineUtils.copy_vcf_record(rec, new_header)

                    df_key = (str(df_row.t_chrom), int(df_row.t_pos), str(df_row.t_ref), str(df_row.t_alt_allele))
                    vcf_key = (str(rec.chrom), int(rec.pos), str(rec.ref), str(rec.alts[0]) if rec.alts else None)
                    if df_key < vcf_key:
                        try:
                            df_row = next(df_iter)
                        except StopIteration:
                            df_row = None
                    elif df_key == vcf_key:
                        new_record.info["xgb_proba"] = round(float(df_row.xgb_proba), 10)
                        try:
                            df_row = next(df_iter)
                        except StopIteration:
                            df_row = None
                        break
                    else:
                        break
                outvcf.write(new_record)
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
