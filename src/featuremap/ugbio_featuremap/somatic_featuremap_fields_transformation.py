import argparse
import logging
import sys
import tempfile
import warnings
from os.path import dirname
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pysam
from ugbio_core.logger import logger
from ugbio_core.vcfbed import vcftools

from ugbio_featuremap import somatic_featuremap_inference_utils
from ugbio_featuremap.featuremap_to_dataframe import vcf_to_parquet
from ugbio_featuremap.featuremap_utils import FeatureMapFields
from ugbio_featuremap.somatic_featuremap_utils import filter_and_annotate_tr

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


info_fields_for_training = ["TR_DISTANCE", "TR_LENGTH", "TR_SEQ_UNIT_LENGTH"]

format_fields_for_training = [
    FeatureMapFields.VAF.value,
    FeatureMapFields.RAW_VAF.value,
    FeatureMapFields.FILT.value,
    FeatureMapFields.DP_FILT.value,
    FeatureMapFields.ADJ_REF_DIFF.value,
    FeatureMapFields.DP_MAPQ60.value,
    FeatureMapFields.DUP.value,
    FeatureMapFields.REV.value,
    FeatureMapFields.SCST.value,
    FeatureMapFields.SCED.value,
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
    "REVERSE_COUNT": ["number of reverse strand reads", "Integer"],
    "FORWARD_COUNT": ["number of forward strand reads", "Integer"],
    "EDIST_MEAN": ["mean value of EDIST", "Float"],
    "EDIST_MAX": ["max value of EDIST", "Float"],
    "EDIST_MIN": ["min value of EDIST", "Float"],
    "RL_MEAN": ["mean value of RL", "Float"],
    "RL_MAX": ["max value of RL", "Float"],
    "RL_MIN": ["min value of RL", "Float"],
    "SCST_NUM_READS": ["number of soft clip start reads", "Integer"],
    "SCED_NUM_READS": ["number of soft clip end reads", "Integer"],
    "MAP0_COUNT": ["number of reads with mapping quality 0", "Integer"],
}
added_info_features = {
    "REF_ALLELE": ["reference allele", "String"],
    "ALT_ALLELE": ["alternative allele", "String"],
}
columns_for_aggregation = [
    FeatureMapFields.MQUAL.value,
    FeatureMapFields.SNVQ.value,
    FeatureMapFields.MAPQ.value,
    FeatureMapFields.EDIST.value,
    FeatureMapFields.RL.value,
]
ORIGINAL_RECORD_INDEX_FIELD = "record_index"

# Sample prefixes for tumor (index 0) and normal (index 1)
TUMOR_PREFIX = "t_"
NORMAL_PREFIX = "n_"


def read_vcf_with_aggregation(vcf_file: str) -> pl.DataFrame:
    """
    Read VCF file using vcf_to_parquet with aggregate mode.

    This is a more efficient alternative to read_merged_tumor_normal_vcf that uses
    bcftools + AWK for aggregation instead of Python lambdas.

    Parameters
    ----------
    vcf_file : str
        Path to the input VCF file (gzipped, indexed).

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with aggregated features and sample-suffixed columns.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "aggregated.parquet"

        # Use vcf_to_parquet with aggregate mode
        # Keep AD column for ALT_READS extraction, expand to AD_0, AD_1
        vcf_to_parquet(
            vcf=vcf_file,
            out=str(parquet_path),
            drop_format={"GT", "X_TCM"},  # Drop unnecessary columns
            list_mode="aggregate",
            expand_columns={"AD": 2},  # Split AD into AD_0 (ref), AD_1 (alt)
        )

        aggregated_df = pl.read_parquet(parquet_path)

    return aggregated_df


def get_sample_names_from_vcf(vcf_file: str) -> tuple[str, str]:
    """
    Get tumor and normal sample names from VCF file.

    Convention: index 0 = tumor, index 1 = normal.

    Parameters
    ----------
    vcf_file : str
        Path to the VCF file.

    Returns
    -------
    tuple[str, str]
        Tuple of (tumor_sample_name, normal_sample_name).
    """
    with pysam.VariantFile(vcf_file) as vcf:
        samples = list(vcf.header.samples)
        if len(samples) < 2:  # noqa: PLR2004
            raise ValueError(f"Expected at least 2 samples in VCF, found {len(samples)}: {samples}")
        return samples[0], samples[1]


def transform_aggregated_df(variants_df: pl.DataFrame, tumor_sample: str, normal_sample: str) -> pd.DataFrame:
    """
    Transform aggregated Polars DataFrame to match the expected format for ML inference.

    This function:
    1. Derives additional columns from aggregation statistics
    2. Renames columns to match the ML model's expected feature names

    Parameters
    ----------
    variants_df : pl.DataFrame
        DataFrame from vcf_to_parquet with aggregate mode.
    tumor_sample : str
        Name of the tumor sample (column suffix).
    normal_sample : str
        Name of the normal sample (column suffix).

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with columns matching ML model expectations.
    """
    # Create sample suffix mappings
    samples = [(tumor_sample, TUMOR_PREFIX), (normal_sample, NORMAL_PREFIX)]

    # Derive columns for each sample
    derived_exprs = []
    rename_map = {}

    for sample_name, prefix in samples:
        s = f"_{sample_name}"  # Original suffix from vcf_to_parquet

        # Derive sum-based columns (sum = mean * count for 0/1 fields)
        # COUNT_DUPLICATE = sum(DUP) = mean(DUP) * count(DUP)
        derived_exprs.append(
            (pl.col(f"DUP_mean{s}") * pl.col(f"DUP_count{s}")).round(0).cast(pl.Int64).alias(f"{prefix}count_duplicate")
        )
        # COUNT_NON_DUPLICATE = count - sum
        derived_exprs.append(
            (pl.col(f"DUP_count{s}") - (pl.col(f"DUP_mean{s}") * pl.col(f"DUP_count{s}")).round(0))
            .cast(pl.Int64)
            .alias(f"{prefix}count_non_duplicate")
        )
        # REVERSE_COUNT = sum(REV)
        derived_exprs.append(
            (pl.col(f"REV_mean{s}") * pl.col(f"REV_count{s}")).round(0).cast(pl.Int64).alias(f"{prefix}reverse_count")
        )
        # FORWARD_COUNT = count - sum
        derived_exprs.append(
            (pl.col(f"REV_count{s}") - (pl.col(f"REV_mean{s}") * pl.col(f"REV_count{s}")).round(0))
            .cast(pl.Int64)
            .alias(f"{prefix}forward_count")
        )
        # PASS_ALT_READS = sum(FILT)
        derived_exprs.append(
            (pl.col(f"FILT_mean{s}") * pl.col(f"FILT_count{s}"))
            .round(0)
            .cast(pl.Int64)
            .alias(f"{prefix}pass_alt_reads")
        )
        # SCST_NUM_READS = count - count_zero (count of non-zero values)
        derived_exprs.append(
            (pl.col(f"SCST_count{s}") - pl.col(f"SCST_count_zero{s}")).cast(pl.Int64).alias(f"{prefix}scst_num_reads")
        )
        # SCED_NUM_READS = count - count_zero
        derived_exprs.append(
            (pl.col(f"SCED_count{s}") - pl.col(f"SCED_count_zero{s}")).cast(pl.Int64).alias(f"{prefix}sced_num_reads")
        )

        # Build rename map for existing columns
        # Aggregation columns (mean/min/max)
        for agg_col in ["MQUAL", "SNVQ", "MAPQ", "EDIST", "RL"]:
            rename_map[f"{agg_col}_mean{s}"] = f"{prefix}{agg_col.lower()}_mean"
            rename_map[f"{agg_col}_min{s}"] = f"{prefix}{agg_col.lower()}_min"
            rename_map[f"{agg_col}_max{s}"] = f"{prefix}{agg_col.lower()}_max"

        # Count zero for MAPQ (MAP0_COUNT)
        rename_map[f"MAPQ_count_zero{s}"] = f"{prefix}map0_count"

        # ALT_READS from AD_1
        rename_map[f"AD_1{s}"] = f"{prefix}alt_reads"

        # Scalar columns
        rename_map[f"DP{s}"] = f"{prefix}dp"
        rename_map[f"VAF{s}"] = f"{prefix}vaf"
        rename_map[f"RAW_VAF{s}"] = f"{prefix}raw_vaf"

    # Apply derived columns
    variants_df = variants_df.with_columns(derived_exprs)

    # Rename columns that exist
    existing_rename = {k: v for k, v in rename_map.items() if k in variants_df.columns}
    variants_df = variants_df.rename(existing_rename)

    # Add common columns for alleles
    variants_df = variants_df.with_columns([pl.col("REF").alias("ref_allele"), pl.col("ALT").alias("alt_allele")])

    # Add TR_DISTANCE for ML model (INFO field, shared between samples)
    if "TR_DISTANCE" in variants_df.columns:
        variants_df = variants_df.with_columns(pl.col("TR_DISTANCE").alias("t_tr_distance"))

    # Rename CHROM and POS columns to match expected format (t_ prefix for VCF writing)
    variants_df = variants_df.with_columns(
        [pl.col("CHROM").alias(f"{TUMOR_PREFIX}chrom"), pl.col("POS").alias(f"{TUMOR_PREFIX}pos")]
    )

    # Handle n_dp fillna with n_ref2 + n_nonref2 (if ref/nonref columns exist)
    # Note: ref/nonref columns from PILEUP require separate processing (TODO)

    # Add record index for VCF writing
    variants_df = variants_df.with_row_index(f"{TUMOR_PREFIX}{ORIGINAL_RECORD_INDEX_FIELD}", offset=1)

    # Convert to pandas for compatibility with existing code
    return variants_df.to_pandas()


def process_sample_columns(df_variants, prefix):  # noqa: C901
    """
    Process columns for a sample with given prefix (t_ or n_).

    This function processes various columns for tumor or normal samples,
    including alternative reads, aggregation features, duplicates, and
    padding reference counts.

    Parameters
    ----------
    df_variants : pd.DataFrame
        DataFrame containing variant information.
    prefix : str
        Sample prefix ('t_' for tumor, 'n_' for normal).

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with processed sample columns.
    """

    def add_agg_features(df, feature_name, prefix):
        df[f"{prefix}{feature_name}_min"] = df[f"{prefix}{feature_name}"].apply(
            lambda x: min(x) if x is not None and len(x) > 0 and None not in x else float("nan")
        )
        df[f"{prefix}{feature_name}_max"] = df[f"{prefix}{feature_name}"].apply(
            lambda x: max(x) if x is not None and len(x) > 0 and None not in x else float("nan")
        )
        df[f"{prefix}{feature_name}_mean"] = df[f"{prefix}{feature_name}"].apply(
            lambda x: sum(x) / len(x) if x is not None and len(x) > 0 and None not in x else float("nan")
        )
        return df

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
        if ref_counts_colname not in df_variants.columns:
            raise ValueError(f"Column {ref_counts_colname} not found in df_variants")

        if df_variants.empty:
            raise ValueError("df_variants is empty, cannot parse padding ref counts")

        if df_variants[ref_counts_colname].isna().all():
            raise ValueError(f"Column {ref_counts_colname} is empty")

        padding_counts_length = len(df_variants[ref_counts_colname].iloc[0])
        # Handle ref_count
        ref_df = pd.DataFrame(
            df_variants[ref_counts_colname].tolist(), columns=[f"{prefix}ref{i}" for i in range(padding_counts_length)]
        )
        for col in ref_df.columns:
            df_variants[col] = ref_df[col].to_numpy()

        # Handle nonref_count
        nonref_df = pd.DataFrame(
            df_variants[non_ref_counts_colname].tolist(),
            columns=[f"{prefix}nonref{i}" for i in range(padding_counts_length)],
        )
        for col in nonref_df.columns:
            df_variants[col] = nonref_df[col].to_numpy()

        return df_variants

    def count_num_sc_reads(df_variants, prefix):
        df_variants[f"{prefix}scst_num_reads"] = df_variants[f"{prefix}scst"].apply(
            lambda x: sum((v is not None) and (v > 0) for v in x) if isinstance(x, (tuple, list)) else np.nan  # noqa: UP038
        )
        df_variants[f"{prefix}sced_num_reads"] = df_variants[f"{prefix}sced"].apply(
            lambda x: sum((v is not None) and (v > 0) for v in x) if isinstance(x, (tuple, list)) else np.nan  # noqa: UP038
        )
        return df_variants

    def count_mapq0_reads(df_variants, prefix):
        df_variants[f"{prefix}map0_count"] = df_variants[f"{prefix}mapq"].apply(
            lambda x: sum(v == 0 for v in x) if x is not None else np.nan
        )
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
        df_variants = add_agg_features(df_variants, f"{colname_lower}", prefix)

    # Process duplicates
    df_variants = parse_is_duplicate(df_variants, f"{prefix}{FeatureMapFields.DUP.value.lower()}", prefix)
    df_variants = parse_padding_ref_counts(
        df_variants,
        f"{prefix}{format_mpileup_fields_for_training[0]}",
        f"{prefix}{format_mpileup_fields_for_training[1]}",
    )
    # Process forward/reverse counts
    df_variants[[f"{prefix}forward_count", f"{prefix}reverse_count"]] = df_variants[f"{prefix}rev"].apply(
        lambda x: pd.Series({"num0": 0, "num1": 0})
        if x is None or (isinstance(x, float) and pd.isna(x))
        else pd.Series({"num0": [v for v in x if v in (0, 1)].count(0), "num1": [v for v in x if v in (0, 1)].count(1)})
    )
    df_variants = count_num_sc_reads(df_variants, prefix)
    df_variants = count_mapq0_reads(df_variants, prefix)

    return df_variants


def df_sfm_fields_transformation(df_variants):  # noqa: C901
    """
    Transform somatic featuremap fields for both tumor and normal samples.

    Processes both tumor ('t_') and normal ('n_') sample columns and adds
    reference and alternative allele information.

    Parameters
    ----------
    df_variants : pd.DataFrame
        DataFrame containing variant information for tumor and normal samples.

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame with processed fields for both samples and
        added allele information.
    """
    # Process both tumor and normal samples
    for prefix in ["t_", "n_"]:
        df_variants = process_sample_columns(df_variants, prefix)
    df_variants["ref_allele"] = [tup[0] for tup in df_variants["t_alleles"]]
    df_variants["alt_allele"] = [tup[1] for tup in df_variants["t_alleles"]]
    df_variants["n_dp"] = df_variants["n_dp"].fillna(df_variants["n_ref2"] + df_variants["n_nonref2"])

    return df_variants


def add_fields_to_header(hdr, added_format_features, added_info_features):
    """
    Add custom FORMAT and INFO fields to VCF header.

    Parameters
    ----------
    hdr : pysam.VariantHeader
        VCF header object to modify.
    added_format_features : dict
        Dictionary of FORMAT fields to add with their descriptions and types.
    added_info_features : dict
        Dictionary of INFO fields to add with their descriptions and types.
    """
    for field in added_format_features:
        if field not in hdr.formats:
            field_type = added_format_features[field][1]
            field_description = added_format_features[field][0]
            hdr.formats.add(field, 1, field_type, field_description)
    for field in added_info_features:
        if field not in hdr.info:
            field_type = added_info_features[field][1]
            field_description = added_info_features[field][0]
            hdr.info.add(field, 1, field_type, field_description)


def process_vcf_records_serially(vcfin, df_variants, hdr, vcfout, write_agg_params):
    """
    Process VCF records serially by iterating through both VCF and DataFrame simultaneously.

    Parameters
    ----------
    vcfin : pysam.VariantFile
        Input VCF file object for reading.
    df_variants : pd.DataFrame
        DataFrame containing variant information with aggregated features.
    hdr : pysam.VariantHeader
        VCF header object for the output file.
    vcfout : pysam.VariantFile
        Output VCF file object for writing.
    write_agg_params : bool
        Whether to write aggregated parameters to the output VCF.
    """

    # Use original record order for processing
    df_variants_sorted = df_variants.sort_values(f"t_{ORIGINAL_RECORD_INDEX_FIELD}").reset_index(drop=True)
    df_iter = iter(df_variants_sorted.itertuples(index=False))
    current_df_record = next(df_iter, None)

    for vcf_row in vcfin:
        vcf_chrom = vcf_row.chrom
        vcf_pos = vcf_row.pos
        vcf_alt = vcf_row.alleles[1] if len(vcf_row.alleles) > 1 else None

        # Skip to matching DataFrame record or process without match
        while current_df_record is not None and (
            current_df_record.t_chrom < vcf_chrom  # case1: next chromosome
            or (current_df_record.t_chrom == vcf_chrom and current_df_record.t_pos < vcf_pos)  # case2: next position
            or (
                current_df_record.t_chrom == vcf_chrom
                and current_df_record.t_pos == vcf_pos
                and current_df_record.alt_allele != vcf_alt  # case3: alt allele for multi-allelic site
            )
        ):
            current_df_record = next(df_iter, None)

        # Check if we have a matching record
        if (
            current_df_record is not None
            and current_df_record.t_chrom == vcf_chrom
            and current_df_record.t_pos == vcf_pos
            and current_df_record.alt_allele == vcf_alt
        ):
            if write_agg_params:
                # Add INFO fields
                for key in added_info_features:
                    vcf_row.info[key.upper()] = getattr(current_df_record, key.lower())

                # Add FORMAT fields
                for key in added_format_features:
                    tumor_value = getattr(current_df_record, f"t_{key.lower()}")
                    normal_value = getattr(current_df_record, f"n_{key.lower()}")

                    if pd.notna(tumor_value):
                        vcf_row.samples[0][key.upper()] = tumor_value
                    else:
                        vcf_row.samples[0][key.upper()] = None

                    if pd.notna(normal_value):
                        vcf_row.samples[1][key.upper()] = normal_value
                    else:
                        vcf_row.samples[1][key.upper()] = None

                # Add XGBoost probability if available
                if "XGB_PROBA" in hdr.info and hasattr(current_df_record, "xgb_proba"):
                    vcf_row.info["XGB_PROBA"] = current_df_record.xgb_proba

        vcfout.write(vcf_row)


def read_merged_tumor_normal_vcf(
    vcf_file: str, custom_info_fields: list[str], fillna_dict: dict[str, object] = None, chrom: str = None
) -> "pd.DataFrame":
    """
    Reads a merged tumor-normal VCF file and returns a concatenated DataFrame with prefixed columns
    for tumor and normal samples.

    Parameters
    ----------
    vcf_file : str
        Path to the VCF file containing both tumor and normal samples.
    custom_info_fields : list of str
        List of custom INFO fields to extract from the VCF.
    fillna_dict : dict of str to object, optional
        Dictionary specifying values to fill missing data for each field.
        Defaults to None.
    chrom : str, optional
        Chromosome to filter the VCF records. If None, all chromosomes are
        included. Defaults to None.

    Returns
    -------
    pd.DataFrame
        A DataFrame with tumor columns prefixed by 't_' and normal columns
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

    # keep original records order
    df_tumor[ORIGINAL_RECORD_INDEX_FIELD] = range(1, len(df_tumor) + 1)
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


def featuremap_fields_aggregation(  # noqa: C901, PLR0915
    somatic_featuremap_vcf_file: str,
    output_vcf: str,
    xgb_model_file: str = None,
    write_agg_params: bool = True,  # noqa: FBT001, FBT002
    verbose: bool = True,  # noqa: FBT001, FBT002
    use_optimized: bool = True,  # noqa: FBT001, FBT002
) -> str:
    """
    Write the vcf file with the aggregated fields and the xgb probability.

    This function processes the entire input VCF file. The input is expected to be
    already filtered and TR-annotated (via filter_and_annotate_tr).

    Parameters
    ----------
    somatic_featuremap_vcf_file : str
        Path to input somatic featuremap VCF file (already filtered and TR-annotated).
    output_vcf : str
        Path to output VCF file with aggregated fields.
    xgb_model_file : str, optional
        Path to XGBoost model file for inference. Defaults to None.
    write_agg_params : bool, optional
        Whether to write aggregated parameters to output. Defaults to True.
    verbose : bool, optional
        Whether to enable verbose logging. Defaults to True.
    use_optimized : bool, optional
        Whether to use the optimized vcf_to_parquet-based reading. Defaults to True.

    Returns
    -------
    str
        Path to the output VCF file with aggregated fields and XGBoost probabilities.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    if use_optimized:
        # Optimized path: use vcf_to_parquet with aggregate mode
        logger.info("Using optimized VCF reading with vcf_to_parquet")

        # Get sample names for column mapping
        tumor_sample, normal_sample = get_sample_names_from_vcf(somatic_featuremap_vcf_file)
        logger.debug(f"Tumor sample: {tumor_sample}, Normal sample: {normal_sample}")

        # Read VCF using vcf_to_parquet with aggregation
        df_polars = read_vcf_with_aggregation(somatic_featuremap_vcf_file)

        if df_polars.is_empty():
            logger.warning(f"No variants found in the input VCF: {somatic_featuremap_vcf_file}")
            # Create an empty VCF with the updated header
            with pysam.VariantFile(somatic_featuremap_vcf_file) as vcfin:
                hdr = vcfin.header
                add_fields_to_header(hdr, added_format_features, added_info_features)
                if xgb_model_file is not None and "XGB_PROBA" not in hdr.info:
                    hdr.info.add("XGB_PROBA", 1, "Float", "XGBoost model predicted probability")
                with pysam.VariantFile(output_vcf, mode="w", header=hdr) as vcfout:
                    pass  # Just create the file with header
            pysam.tabix_index(output_vcf, preset="vcf", min_shift=0, force=True)
            return output_vcf

        # Transform aggregated DataFrame to match expected format
        df_variants = transform_aggregated_df(df_polars, tumor_sample, normal_sample)
        logger.info(f"Transformed DataFrame with {len(df_variants)} variants")
    else:
        # Legacy path: use original pysam-based reading with pandas lambdas
        logger.info("Using legacy VCF reading with pysam")
        custom_info_fields = (
            format_fields_for_training
            + format_mpileup_fields_for_training
            + info_fields_for_training
            + columns_for_aggregation
        )
        df_variants = read_merged_tumor_normal_vcf(somatic_featuremap_vcf_file, custom_info_fields=custom_info_fields)
        if len(df_variants) == 0:
            logger.warning(f"No variants found in the input VCF: {somatic_featuremap_vcf_file}")
            # Create an empty VCF with the updated header
            with pysam.VariantFile(somatic_featuremap_vcf_file) as vcfin:
                hdr = vcfin.header
                add_fields_to_header(hdr, added_format_features, added_info_features)
                if xgb_model_file is not None and "XGB_PROBA" not in hdr.info:
                    hdr.info.add("XGB_PROBA", 1, "Float", "XGBoost model predicted probability")
                with pysam.VariantFile(output_vcf, mode="w", header=hdr) as vcfout:
                    pass  # Just create the file with header
            pysam.tabix_index(output_vcf, preset="vcf", min_shift=0, force=True)
            return output_vcf
        df_variants = df_sfm_fields_transformation(df_variants)

    # XGBoost inference
    if xgb_model_file is not None:
        xgb_clf_es = somatic_featuremap_inference_utils.load_model(xgb_model_file)
        model_features = xgb_clf_es.get_booster().feature_names
        logger.info(f"loaded model. model features: {model_features}")
        df_variants["xgb_proba"] = somatic_featuremap_inference_utils.predict(xgb_clf_es, df_variants)

    # Write df_variants to parquet file
    parquet_output = output_vcf.replace(".vcf.gz", "_featuremap.parquet")
    df_variants.to_parquet(parquet_output, index=False)
    logger.info(f"Written feature map dataframe to {parquet_output}")

    # Serial processing to avoid expensive lookups for each record
    with pysam.VariantFile(somatic_featuremap_vcf_file) as vcfin:
        hdr = vcfin.header
        add_fields_to_header(hdr, added_format_features, added_info_features)
        if xgb_model_file is not None and "XGB_PROBA" not in hdr.info:
            hdr.info.add("XGB_PROBA", 1, "Float", "XGBoost model predicted probability")
        with pysam.VariantFile(output_vcf, mode="w", header=hdr) as vcfout:
            process_vcf_records_serially(vcfin, df_variants, hdr, vcfout, write_agg_params)
    pysam.tabix_index(output_vcf, preset="vcf", min_shift=0, force=True)
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
        "-g",
        "--genome_file",
        type=str,
        required=True,
        help="""Genome FASTA index (.fai) file""",
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
        "--disable_verbose",
        action="store_false",
        default=True,
        help="""Disable verbose output and debug messages. When used, verbose mode is turned off
                (default: verbose enabled)""",
    )
    return parser.parse_args(argv[1:])


def run(argv):
    """
    Add aggregated parameters and xgb probability to the featuremap pileup vcf file.

    This script processes pre-chunked somatic featuremap VCF files. Parallelization
    is handled externally by the pipeline - this script processes the entire input
    file in a single pass for optimal performance.

    Parameters
    ----------
    argv : list of str
        Command line arguments including the script name.
    """
    args_in = __parse_args(argv)

    # Ensure output has .vcf.gz suffix
    output_vcf = args_in.output_vcf
    if not output_vcf.endswith(".vcf.gz"):
        logger.debug("adding .vcf.gz suffix to the output vcf file")
        output_vcf = output_vcf + ".vcf.gz"

    # Unified preprocessing: filter PASS variants AND add tandem repeat features
    # This is more efficient than doing them separately since we only process
    # PASS variants through TR annotation
    out_dir = dirname(output_vcf)
    sfm_filtered_with_tr = filter_and_annotate_tr(
        args_in.somatic_featuremap,
        args_in.ref_tr_file,
        args_in.genome_file,
        out_dir,
        filter_string=args_in.filter_string,
    )

    # Process entire file directly (already filtered and TR-annotated)
    featuremap_fields_aggregation(
        sfm_filtered_with_tr,
        output_vcf,
        xgb_model_file=args_in.xgb_model_file,
        verbose=args_in.disable_verbose,
    )


def main():
    """
    Main entry point for the script.

    Calls run() with command line arguments from sys.argv.
    """
    run(sys.argv)


if __name__ == "__main__":
    main()
