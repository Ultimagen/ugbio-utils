import argparse
import logging
import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd
import polars as pl
import pysam
from ugbio_core.logger import logger

from ugbio_featuremap import somatic_featuremap_inference_utils
from ugbio_featuremap.featuremap_to_dataframe import vcf_to_parquet
from ugbio_featuremap.somatic_featuremap_utils import filter_and_annotate_tr

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# REQUIRED COLUMNS FOR ML INFERENCE
# These are the columns needed from the VCF to compute the model's expected features.
# =============================================================================

# INFO fields required for inference
REQUIRED_INFO_FIELDS: set[str] = {
    "TR_DISTANCE",  # Tandem repeat distance (added by TR annotation step)
}

# FORMAT fields required for inference (per-sample fields)
# These are used directly or for deriving aggregated features
REQUIRED_FORMAT_FIELDS: set[str] = {
    "DP",  # Read depth -> t_dp, n_dp
    "VAF",  # Variant allele frequency -> t_vaf, n_vaf
    "RAW_VAF",  # Raw VAF -> t_raw_vaf, n_raw_vaf
    "AD",  # Allelic depths -> AD_1 for alt_reads
    "MQUAL",  # Mapping quality per read -> mean/min/max aggregations
    "SNVQ",  # SNV quality per read -> mean/min/max aggregations
    "MAPQ",  # Mapping quality (for count_zero -> map0_count)
    "EDIST",  # Edit distance -> mean/min/max aggregations
    "RL",  # Read length -> mean/min/max aggregations
    "DUP",  # Duplicate flag -> count_duplicate, count_non_duplicate
    "REV",  # Reverse strand flag -> reverse_count, forward_count
    "FILT",  # Filter flag -> pass_alt_reads
    "SCST",  # Soft clip start -> scst_num_reads (count non-zero)
    "SCED",  # Soft clip end -> sced_num_reads (count non-zero)
    # PILEUP columns for ref0-4 / nonref0-4 calculations
    "PILEUP_A_L2",
    "PILEUP_A_L1",
    "PILEUP_A_C",
    "PILEUP_A_R1",
    "PILEUP_A_R2",
    "PILEUP_C_L2",
    "PILEUP_C_L1",
    "PILEUP_C_C",
    "PILEUP_C_R1",
    "PILEUP_C_R2",
    "PILEUP_G_L2",
    "PILEUP_G_L1",
    "PILEUP_G_C",
    "PILEUP_G_R1",
    "PILEUP_G_R2",
    "PILEUP_T_L2",
    "PILEUP_T_L1",
    "PILEUP_T_C",
    "PILEUP_T_R1",
    "PILEUP_T_R2",
    "PILEUP_DEL_L2",
    "PILEUP_DEL_L1",
    "PILEUP_DEL_C",
    "PILEUP_DEL_R1",
    "PILEUP_DEL_R2",
    "PILEUP_INS_L2",
    "PILEUP_INS_L1",
    "PILEUP_INS_C",
    "PILEUP_INS_R1",
    "PILEUP_INS_R2",
}

# =============================================================================
# VCF OUTPUT FIELD DEFINITIONS
# =============================================================================

# TODO: I'm not sure if we stil need ADDED_FORMAT_FEATURES and ADDED_INFO_FEATURES since they are added to the parquet and not the output VCF
ADDED_FORMAT_FEATURES = {
    "ALT_READS": ["number of supporting reads for the alternative allele", "Integer"],
    "PASS_ALT_READS": ["number of passed supporting reads for the alternative allele", "Integer"],
    "MQUAL_MEAN": ["mean value of MQUAL", "Float"],
    "SNVQ_MEAN": ["mean value of SNVQ", "Float"],
    "MQUAL_MAX": ["max value of MQUAL", "Float"],
    "SNVQ_MAX": ["max value of SNVQ", "Float"],
    "MQUAL_MIN": ["min value of MQUAL", "Float"],
    "SNVQ_MIN": ["min value of SNVQ", "Float"],
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

ADDED_INFO_FEATURES = {
    "REF_ALLELE": ["reference allele", "String"],
    "ALT_ALLELE": ["alternative allele", "String"],
}

# Sample prefixes for tumor (index 0) and normal (index 1)
TUMOR_PREFIX = "t_"
NORMAL_PREFIX = "n_"


# =============================================================================
# STEP 1: Filter for PASS + Add Tandem Repeat
# This is implemented in somatic_featuremap_utils.filter_and_annotate_tr()
# =============================================================================


# =============================================================================
# STEP 2: Convert VCF to DataFrame + Post-Processing
# =============================================================================


def get_columns_to_drop_from_vcf(vcf_path: Path) -> tuple[set[str], set[str]]:
    """
    Determine which INFO and FORMAT columns to drop based on required fields.

    Reads the VCF header to get all available columns, then subtracts the
    required columns to get the list of columns to drop.

    Parameters
    ----------
    vcf_path : Path
        Path to the VCF file.

    Returns
    -------
    tuple[set[str], set[str]]
        Tuple of (drop_info, drop_format) sets.
    """
    with pysam.VariantFile(str(vcf_path)) as vcf:
        all_info_fields = set(vcf.header.info.keys())
        all_format_fields = set(vcf.header.formats.keys())

    drop_info = all_info_fields - REQUIRED_INFO_FIELDS
    drop_format = all_format_fields - REQUIRED_FORMAT_FIELDS

    logger.debug(f"Dropping INFO fields: {drop_info}")
    logger.debug(f"Dropping FORMAT fields: {drop_format}")

    return drop_info, drop_format


def read_vcf_with_aggregation(vcf_path: Path) -> pl.DataFrame:
    """
    Read VCF file into polars dataframe with column aggregations.

    Only keeps columns required for ML inference (defined in REQUIRED_INFO_FIELDS
    and REQUIRED_FORMAT_FIELDS). Other columns are dropped for efficiency.

    Parameters
    ----------
    vcf_path : Path
        Path to the input VCF file (gzipped, indexed).

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with aggregated features and sample-suffixed columns.
    """
    # Compute which columns to drop based on required fields
    drop_info, drop_format = get_columns_to_drop_from_vcf(vcf_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = Path(tmpdir) / "aggregated.parquet"

        vcf_to_parquet(
            vcf=str(vcf_path),
            out=str(parquet_path),
            drop_info=drop_info,
            drop_format=drop_format,
            list_mode="aggregate",
            expand_columns={"AD": 2},  # Split AD into AD_0 (ref), AD_1 (alt)
        )

        logger.info(f"Read aggregated dataframe from parquet file: {parquet_path}")
        aggregated_df = pl.read_parquet(parquet_path)

    return aggregated_df


def get_sample_names_from_vcf(vcf_path: Path) -> tuple[str, str]:
    """
    Get tumor and normal sample names from VCF file.

    Convention: index 0 = tumor, index 1 = normal.

    Parameters
    ----------
    vcf_path : Path
        Path to the VCF file.

    Returns
    -------
    tuple[str, str]
        Tuple of (tumor_sample_name, normal_sample_name).
    """
    with pysam.VariantFile(str(vcf_path)) as vcf:
        samples = list(vcf.header.samples)
        if len(samples) < 2:  
            raise ValueError(f"Expected at least 2 samples in VCF, found {len(samples)}: {samples}")
        return samples[0], samples[1]


# PILEUP position mapping: L2→ref0, L1→ref1, C→ref2, R1→ref3, R2→ref4
PILEUP_POSITIONS = ["L2", "L1", "C", "R1", "R2"]
PILEUP_BASES = ["A", "C", "G", "T"]
PILEUP_INDELS = ["DEL", "INS"]


def calculate_ref_nonref_columns(variants_df: pl.DataFrame, sample_suffix: str, output_prefix: str) -> pl.DataFrame:
    """
    Calculate ref0-4 and nonref0-4 columns from PILEUP columns.

    For each position (L2, L1, C, R1, R2) → (ref0, ref1, ref2, ref3, ref4):
    - ref{i} = PILEUP_{REF}_{pos}_{sample}
    - nonref{i} = sum of PILEUP_{non-REF bases}_{pos} + PILEUP_{DEL}_{pos} + PILEUP_{INS}_{pos}

    Parameters
    ----------
    variants_df : pl.DataFrame
        DataFrame with PILEUP columns.
    sample_suffix : str
        Sample suffix from vcf_to_parquet (e.g., "_Pa_46_FreshFrozen").
    output_prefix : str
        Output column prefix ("t_" or "n_").

    Returns
    -------
    pl.DataFrame
        DataFrame with ref0-4 and nonref0-4 columns added.
    """
    ref_nonref_exprs = []

    for i, pos in enumerate(PILEUP_POSITIONS):
        # Build ref column: select the PILEUP column matching the REF allele
        ref_expr = (
            pl.when(pl.col("REF") == "A")
            .then(pl.col(f"PILEUP_A_{pos}{sample_suffix}"))
            .when(pl.col("REF") == "C")
            .then(pl.col(f"PILEUP_C_{pos}{sample_suffix}"))
            .when(pl.col("REF") == "G")
            .then(pl.col(f"PILEUP_G_{pos}{sample_suffix}"))
            .when(pl.col("REF") == "T")
            .then(pl.col(f"PILEUP_T_{pos}{sample_suffix}"))
            .otherwise(pl.lit(0))
            .fill_null(0)
            .alias(f"{output_prefix}ref{i}")
        )
        ref_nonref_exprs.append(ref_expr)

        # Build nonref column: sum of non-REF bases + DEL + INS
        nonref_components = []

        # Add non-REF base counts (exclude the REF base)
        for base in PILEUP_BASES:
            col_name = f"PILEUP_{base}_{pos}{sample_suffix}"
            if col_name in variants_df.columns:
                nonref_components.append(
                    pl.when(pl.col("REF") != base).then(pl.col(col_name).fill_null(0)).otherwise(pl.lit(0))
                )

        # Add DEL and INS counts (always included in nonref)
        for indel in PILEUP_INDELS:
            col_name = f"PILEUP_{indel}_{pos}{sample_suffix}"
            if col_name in variants_df.columns:
                nonref_components.append(pl.col(col_name).fill_null(0))

        if nonref_components:
            nonref_expr = pl.sum_horizontal(nonref_components).alias(f"{output_prefix}nonref{i}")
            ref_nonref_exprs.append(nonref_expr)

    if ref_nonref_exprs:
        variants_df = variants_df.with_columns(ref_nonref_exprs)

    return variants_df


def calculate_pileup_features(variants_df: pl.DataFrame, tumor_sample: str, normal_sample: str) -> pl.DataFrame:
    """
    Calculate PILEUP-based ref0-4 and nonref0-4 features for both samples.

    Parameters
    ----------
    variants_df : pl.DataFrame
        DataFrame with PILEUP columns from vcf_to_parquet.
    tumor_sample : str
        Tumor sample name.
    normal_sample : str
        Normal sample name.

    Returns
    -------
    pl.DataFrame
        DataFrame with ref0-4, nonref0-4 columns for both t_ and n_ prefixes.
    """
    # Calculate for tumor sample
    variants_df = calculate_ref_nonref_columns(variants_df, f"_{tumor_sample}", TUMOR_PREFIX)

    # Calculate for normal sample
    variants_df = calculate_ref_nonref_columns(variants_df, f"_{normal_sample}", NORMAL_PREFIX)

    return variants_df


def aggregated_df_post_processing(variants_df: pl.DataFrame, tumor_sample: str, normal_sample: str) -> pd.DataFrame:
    """
    Post-process the aggregated Polars DataFrame to include all required features by ML inference.
    E.g.  add Sum, count non zero and add ref/nonref columns for PILEUP columns, etc.

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

    # TODO: I'm not sure if we still need this since we calcaulte the ref_nonref differntly now
    # Handle n_dp fillna with n_ref2 + n_nonref2 (if ref/nonref columns exist)
    if "n_dp" in variants_df.columns and "n_ref2" in variants_df.columns and "n_nonref2" in variants_df.columns:
        variants_df = variants_df.with_columns(
            pl.when(pl.col("n_dp").is_null())
            .then(pl.col("n_ref2") + pl.col("n_nonref2"))
            .otherwise(pl.col("n_dp"))
            .alias("n_dp")
        )

    # TODO: don't convert to pandas, keep as polars dataframe once model compatibility verified
    # Convert to pandas for compatibility with existing code
    return variants_df.to_pandas()


# =============================================================================
# STEP 3: Prepare Data + Run Classifier
# =============================================================================


def run_classifier(
    df_variants: pd.DataFrame,
    xgb_model_path: Path,
) -> pd.DataFrame:
    """
    Run XGBoost classifier on the prepared DataFrame.

    Parameters
    ----------
    df_variants : pd.DataFrame
        DataFrame with features prepared for ML inference.
    xgb_model_path : Path
        Path to the XGBoost model file.

    Returns
    -------
    pd.DataFrame
        DataFrame with xgb_proba column added.
    """
    xgb_clf = somatic_featuremap_inference_utils.load_xgb_model(xgb_model_path)
    model_features = xgb_clf.get_booster().feature_names
    logger.info(f"Loaded model with features: {model_features}")

    df_variants["xgb_proba"] = somatic_featuremap_inference_utils.predict(xgb_clf, df_variants)
    return df_variants


# =============================================================================
# STEP 4: Write Output VCF
# Implemented in write_enhanced_vcf() above
# =============================================================================

# TODO: should we only add the xgb_proba field to the output VCF?
def add_fields_to_header(hdr: pysam.VariantHeader) -> None:
    """
    Add custom FORMAT and INFO fields to VCF header.

    Uses the global ADDED_FORMAT_FEATURES and ADDED_INFO_FEATURES dictionaries.

    Parameters
    ----------
    hdr : pysam.VariantHeader
        VCF header object to modify.
    """
    for field, (description, field_type) in ADDED_FORMAT_FEATURES.items():
        if field not in hdr.formats:
            hdr.formats.add(field, 1, field_type, description)
    for field, (description, field_type) in ADDED_INFO_FEATURES.items():
        if field not in hdr.info:
            hdr.info.add(field, 1, field_type, description)


def _is_valid_value(value: object) -> bool:
    """Check if a value is valid (not None and not NaN)."""
    if value is None:
        return False
    try:
        return bool(pd.notna(value))
    except (ValueError, TypeError):
        return True


def _build_variant_lookup(df_variants: pd.DataFrame) -> dict[tuple[str, int, str], dict]:
    """Build lookup dictionary from DataFrame for O(1) access during VCF writing."""
    lookup: dict[tuple[str, int, str], dict] = {}
    for _, row in df_variants.iterrows():
        chrom = row["t_chrom"]
        pos = row["t_pos"]
        alt = row["alt_allele"]
        key = (str(chrom), int(pos), str(alt))
        lookup[key] = row.to_dict()
    return lookup


def _add_record_fields(vcf_row: pysam.VariantRecord, df_record: dict, write_agg_params: bool) -> None:  # noqa: FBT001
    """Add aggregated fields and XGBoost probability to a VCF record."""
    if write_agg_params:
        # Add INFO fields
        for field_name in ADDED_INFO_FEATURES:
            value = df_record.get(field_name.lower())
            if _is_valid_value(value):
                vcf_row.info[field_name] = value

        # Add FORMAT fields for both samples
        for field_name in ADDED_FORMAT_FEATURES:
            tumor_value = df_record.get(f"t_{field_name.lower()}")
            normal_value = df_record.get(f"n_{field_name.lower()}")

            vcf_row.samples[0][field_name] = tumor_value if _is_valid_value(tumor_value) else None
            vcf_row.samples[1][field_name] = normal_value if _is_valid_value(normal_value) else None

    # Add XGBoost probability
    xgb_proba = df_record.get("xgb_proba")
    if _is_valid_value(xgb_proba):
        vcf_row.info["XGB_PROBA"] = xgb_proba


def write_enhanced_vcf(
    input_vcf_path: Path,
    output_vcf_path: Path,
    df_variants: pd.DataFrame,
    write_agg_params: bool = True,  # noqa: FBT001, FBT002
) -> None:
    """
    Write enhanced VCF with aggregated features and XGBoost probability.

    Implements a Batch write with lookup.
    Creates a lookup dictionary for O(1) access, then iterates through VCF records
    and adds the computed features.

    Parameters
    ----------
    input_vcf_path : Path
        Path to the input VCF file.
    output_vcf_path : Path
        Path to the output VCF file.
    df_variants : pd.DataFrame
        DataFrame containing variant information with aggregated features and xgb_proba.
    write_agg_params : bool, optional
        Whether to write aggregated parameters to the output VCF. Defaults to True.
    """
    lookup = _build_variant_lookup(df_variants)

    with pysam.VariantFile(str(input_vcf_path)) as vcfin:
        hdr = vcfin.header
        add_fields_to_header(hdr)
        if "XGB_PROBA" not in hdr.info:
            hdr.info.add("XGB_PROBA", 1, "Float", "XGBoost model predicted probability")

        with pysam.VariantFile(str(output_vcf_path), mode="w", header=hdr) as vcfout:
            for vcf_row in vcfin:
                alleles = vcf_row.alleles
                vcf_alt = alleles[1] if alleles is not None and len(alleles) > 1 else None

                if vcf_alt is not None:
                    key = (vcf_row.chrom, vcf_row.pos, vcf_alt)
                    df_record = lookup.get(key)

                    if df_record is not None:
                        _add_record_fields(vcf_row, df_record, write_agg_params)

                vcfout.write(vcf_row)

    pysam.tabix_index(str(output_vcf_path), preset="vcf", min_shift=0, force=True)

# =============================================================================
# MAIN ORCHESTRATION FUNCTION
# =============================================================================


def __parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="somatic featuremap fields transformation",
        description=run.__doc__,
    )
    parser.add_argument(
        "-sfm",
        "--somatic_featuremap",
        type=Path,
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
        type=Path,
        required=True,
        help="""Output pileup vcf file""",
    )
    parser.add_argument(
        "-g",
        "--genome_file",
        type=Path,
        required=True,
        help="""Genome FASTA index (.fai) file""",
    )
    parser.add_argument(
        "-ref_tr",
        "--ref_tr_file",
        type=Path,
        required=True,
        help="""Reference tandem repeat file in BED format""",
    )
    parser.add_argument(
        "-xgb_model",
        "--xgb_model_file",
        type=Path,
        required=True,
        help="""XGBoost model file for inference""",
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


def run(argv: list[str]) -> None:
    args_in = __parse_args(argv)

    # Ensure output has .vcf.gz suffix
    output_vcf = args_in.output_vcf
    if not str(output_vcf).endswith(".vcf.gz"):
        logger.debug("adding .vcf.gz suffix to the output vcf file")
        output_vcf = output_vcf.with_suffix(".vcf.gz")

    # Use output directory for temporary files
    out_dir = output_vcf.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    somatic_featuremap_classifier(
        somatic_featuremap_vcf_path=args_in.somatic_featuremap, 
        ref_tr_file=args_in.ref_tr_file, 
        genome_file=args_in.genome_file, 
        out_dir=out_dir, 
        filter_string=args_in.filter_string, 
        xgb_model_path=args_in.xgb_model_file, 
        output_vcf=output_vcf, 
        write_agg_params=args_in.write_agg_params, 
        verbose=args_in.verbose,
    )


def somatic_featuremap_classifier(
    somatic_featuremap_vcf_path: Path, 
    ref_tr_file: Path, 
    genome_file: Path, 
    out_dir: Path, 
    filter_string: str, 
    xgb_model_path: Path, 
    output_vcf: Path, 
    *,
    write_agg_params: bool = False,  
    verbose: bool = False,  
) -> Path:
    """
    Classify somatic featuremap variants using XGBoost model.

    Steps:
    1. Filter VCF and add TR annotations
    2. Convert VCF to dataframe and add transformations such asaggreagtions and PILEUP-based ref/nonref features
    3. Run classifier and add XGBoost probability on the dataframe
    4. Write enhanced VCF with XGBoost probability
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    logger.info("Step 1: Filtering VCF and adding TR annotations")
    sfm_filtered_with_tr = filter_and_annotate_tr(
        somatic_featuremap_vcf_path,
        ref_tr_file,
        genome_file,
        out_dir,
        filter_string=filter_string,
    )

    logger.info("Step 2: Converting VCF to dataframe and adding transformations")
    df_polars = read_vcf_with_aggregation(sfm_filtered_with_tr)

    if df_polars.is_empty():
        # TODO: handle better
        logger.warning(f"No variants found in the input VCF: {somatic_featuremap_vcf_path}")
    
    tumor_sample, normal_sample = get_sample_names_from_vcf(somatic_featuremap_vcf_path)

    logger.info("Calculating PILEUP-based ref/nonref features")
    df_polars = calculate_pileup_features(df_polars, tumor_sample, normal_sample)
    logger.info("Post-processing the aggregated dataframe")
    df_variants = aggregated_df_post_processing(df_polars, tumor_sample, normal_sample)

    logger.info("Step 3: Running the classifier")
    df_variants = run_classifier(df_variants, xgb_model_path)

    logger.info("Step 4: Writing enhanced VCF")
    write_enhanced_vcf(somatic_featuremap_vcf_path, output_vcf, df_variants, write_agg_params)
    logger.info(f"Output VCF written to: {output_vcf}")

    return output_vcf


def main():
    """
    Main entry point for the script.

    Calls run() with command line arguments from sys.argv.
    """
    run(sys.argv)


if __name__ == "__main__":
    main()
