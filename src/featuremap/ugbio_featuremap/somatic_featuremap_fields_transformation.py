import logging
import warnings
from pathlib import Path

import cyclopts
import pandas as pd
import polars as pl
import pysam
from rich.console import Console
from ugbio_core.logger import logger

from ugbio_featuremap import somatic_featuremap_inference_utils
from ugbio_featuremap.featuremap_to_dataframe import vcf_to_parquet
from ugbio_featuremap.somatic_featuremap_utils import (
    PILEUP_CONFIG,
    TR_CONFIG,
    filter_and_annotate_tr,
)

app = cyclopts.App(
    help="Classify somatic featuremap variants using XGBoost model.",
)
console = Console(stderr=True)

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# REQUIRED COLUMNS FOR ML INFERENCE
# These are the columns needed from the VCF to compute the model's expected features.
# =============================================================================

# INFO fields required for inference (includes TR_DISTANCE field added by annotation step)
REQUIRED_INFO_FIELDS: set[str] = {TR_CONFIG.distance_field_id}

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
    *PILEUP_CONFIG.get_all_format_fields(),
}

# =============================================================================
# VCF OUTPUT FIELD DEFINITIONS
# =============================================================================

# TODO: I'm not sure if we still need ADDED_FORMAT_FEATURES and ADDED_INFO_FEATURES
# since they are added to the parquet and not the output VCF
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


def read_vcf_with_aggregation(vcf_path: Path, output_parquet_path: Path) -> pl.DataFrame:
    """
    Read VCF file into polars dataframe with column aggregations.

    Only keeps columns required for ML inference (defined in REQUIRED_INFO_FIELDS
    and REQUIRED_FORMAT_FIELDS). Other columns are dropped for efficiency.

    Parameters
    ----------
    vcf_path : Path
        Path to the input VCF file (gzipped, indexed).
    output_parquet_path : Path
        Path to save the aggregated parquet file.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with aggregated features and sample-suffixed columns.
    """
    # Compute which columns to drop based on required fields
    drop_info, drop_format = get_columns_to_drop_from_vcf(vcf_path)

    vcf_to_parquet(
        vcf=str(vcf_path),
        out=str(output_parquet_path),
        drop_info=drop_info,
        drop_format=drop_format,
        list_mode="aggregate",
        expand_columns={"AD": 2},  # Split AD into AD_0 (ref), AD_1 (alt)
    )

    logger.info(f"Read aggregated dataframe from parquet file: {output_parquet_path}")
    aggregated_df = pl.read_parquet(output_parquet_path)

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
        if len(samples) < 2:  # noqa: PLR2004
            raise ValueError(f"Expected at least 2 samples in VCF, found {len(samples)}: {samples}")
        return samples[0], samples[1]


def calculate_ref_nonref_columns(variants_df: pl.DataFrame, sample_suffix: str) -> pl.DataFrame:
    """
    Calculate ref0-4 and nonref0-4 columns from PILEUP columns.

    For each position (L2, L1, C, R1, R2) → (ref0, ref1, ref2, ref3, ref4):
    - ref{i} = PILEUP_{REF}_{pos}_{sample}
    - nonref{i} = sum of PILEUP_{non-REF bases}_{pos} + PILEUP_{DEL}_{pos} + PILEUP_{INS}_{pos}

    Output columns follow sample suffix convention (e.g., ref0_Pa_46_FreshFrozen).

    Parameters
    ----------
    variants_df : pl.DataFrame
        DataFrame with PILEUP columns.
    sample_suffix : str
        Sample suffix from vcf_to_parquet (e.g., "_Pa_46_FreshFrozen").

    Returns
    -------
    pl.DataFrame
        DataFrame with ref0-4 and nonref0-4 columns added (with sample suffix).
    """
    ref_nonref_exprs = []

    for i, pos in enumerate(PILEUP_CONFIG.positions):
        # Build ref column: select the PILEUP column matching the REF allele
        # TODO: use PILEUP_CONFIG.bases instead of hardcoding the bases
        ref_expr = (
            pl.when(pl.col("REF") == "A")
            .then(pl.col(PILEUP_CONFIG.get_column_name("A", pos, sample_suffix)))
            .when(pl.col("REF") == "C")
            .then(pl.col(PILEUP_CONFIG.get_column_name("C", pos, sample_suffix)))
            .when(pl.col("REF") == "G")
            .then(pl.col(PILEUP_CONFIG.get_column_name("G", pos, sample_suffix)))
            .when(pl.col("REF") == "T")
            .then(pl.col(PILEUP_CONFIG.get_column_name("T", pos, sample_suffix)))
            .otherwise(pl.lit(0))
            .fill_null(0)
            .alias(f"ref{i}{sample_suffix}")
        )
        ref_nonref_exprs.append(ref_expr)

        # Build nonref column: sum of non-REF bases + DEL + INS
        nonref_components = []

        # Add non-REF base counts (exclude the REF base)
        for base in PILEUP_CONFIG.bases:
            col_name = PILEUP_CONFIG.get_column_name(base, pos, sample_suffix)
            if col_name in variants_df.columns:
                nonref_components.append(
                    pl.when(pl.col("REF") != base).then(pl.col(col_name).fill_null(0)).otherwise(pl.lit(0))
                )

        # Add DEL and INS counts (always included in nonref)
        for indel in PILEUP_CONFIG.indels:
            col_name = PILEUP_CONFIG.get_column_name(indel, pos, sample_suffix)
            if col_name in variants_df.columns:
                nonref_components.append(pl.col(col_name).fill_null(0))

        if nonref_components:
            nonref_expr = pl.sum_horizontal(nonref_components).alias(f"nonref{i}{sample_suffix}")
            ref_nonref_exprs.append(nonref_expr)

    if ref_nonref_exprs:
        variants_df = variants_df.with_columns(ref_nonref_exprs)

    return variants_df


def calculate_pileup_features(variants_df: pl.DataFrame, tumor_sample: str, normal_sample: str) -> pl.DataFrame:
    """
    Calculate PILEUP-based ref0-4 and nonref0-4 features for both samples.

    Output columns use sample suffix convention (e.g., ref0_Pa_46_FreshFrozen).

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
        DataFrame with ref0-4, nonref0-4 columns for both samples (with sample suffix).
    """
    # Calculate for tumor sample
    variants_df = calculate_ref_nonref_columns(variants_df, f"_{tumor_sample}")

    # Calculate for normal sample
    variants_df = calculate_ref_nonref_columns(variants_df, f"_{normal_sample}")

    return variants_df


def aggregated_df_post_processing(variants_df: pl.DataFrame, samples: list[str]) -> pl.DataFrame:
    """
    Post-process the aggregated Polars DataFrame to include all required features by ML inference.
    Derives additional columns from aggregation statistics (with sample suffix)
    E.g.  add Sum, count non zero and add ref/nonref columns for PILEUP columns, etc.


    Parameters
    ----------
    variants_df : pl.DataFrame
        DataFrame from vcf_to_parquet with aggregate mode.
    samples : list[str]
        List of sample names.

    Returns
    -------
    pd.DataFrame
        Pandas DataFrame with columns matching ML model expectations.
    """

    # Derive columns for each sample (using sample suffix convention)
    derived_exprs = []

    for sample_name in samples:
        s = f"_{sample_name}"  # Original suffix from vcf_to_parquet

        # Derive sum-based columns (sum = mean * count for 0/1 fields)
        # COUNT_DUPLICATE = sum(DUP) = mean(DUP) * count(DUP)
        derived_exprs.append(
            (pl.col(f"DUP_mean{s}") * pl.col(f"DUP_count{s}")).round(0).cast(pl.Int64).alias(f"count_duplicate{s}")
        )
        # COUNT_NON_DUPLICATE = count - sum
        derived_exprs.append(
            (pl.col(f"DUP_count{s}") - (pl.col(f"DUP_mean{s}") * pl.col(f"DUP_count{s}")).round(0))
            .cast(pl.Int64)
            .alias(f"count_non_duplicate{s}")
        )
        # REVERSE_COUNT = sum(REV)
        derived_exprs.append(
            (pl.col(f"REV_mean{s}") * pl.col(f"REV_count{s}")).round(0).cast(pl.Int64).alias(f"reverse_count{s}")
        )
        # FORWARD_COUNT = count - sum
        derived_exprs.append(
            (pl.col(f"REV_count{s}") - (pl.col(f"REV_mean{s}") * pl.col(f"REV_count{s}")).round(0))
            .cast(pl.Int64)
            .alias(f"forward_count{s}")
        )
        # PASS_ALT_READS = sum(FILT)
        derived_exprs.append(
            (pl.col(f"FILT_mean{s}") * pl.col(f"FILT_count{s}")).round(0).cast(pl.Int64).alias(f"pass_alt_reads{s}")
        )
        # SCST_NUM_READS = count - count_zero (count of non-zero values)
        derived_exprs.append(
            (pl.col(f"SCST_count{s}") - pl.col(f"SCST_count_zero{s}")).cast(pl.Int64).alias(f"scst_num_reads{s}")
        )
        # SCED_NUM_READS = count - count_zero
        derived_exprs.append(
            (pl.col(f"SCED_count{s}") - pl.col(f"SCED_count_zero{s}")).cast(pl.Int64).alias(f"sced_num_reads{s}")
        )

    # Apply derived columns
    variants_df = variants_df.with_columns(derived_exprs)

    return variants_df


def rename_cols_for_model(variants_df: pl.DataFrame, samples: list[str]) -> pl.DataFrame:
    """
    Rename the dataframe columns to match the model expected features.

    Parameters
    ----------
    variants_df : pl.DataFrame
        DataFrame with sample-suffixed columns.
    samples : list[tuple[str, str]]
        List of (sample_name, prefix) tuples.
    """

    # Build rename map: convert all sample-suffixed columns to t_/n_ prefix convention
    rename_map = {}

    # rename INFO fields
    for info_field in REQUIRED_INFO_FIELDS:
        rename_map[info_field] = f"{TUMOR_PREFIX}{info_field.lower()}"

    # rename FORMAT fields
    for sample_name, prefix in [(samples[0], TUMOR_PREFIX), (samples[1], NORMAL_PREFIX)]:
        s = f"_{sample_name}"

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

        # Derived columns in post_processing
        rename_map[f"count_duplicate{s}"] = f"{prefix}count_duplicate"
        rename_map[f"count_non_duplicate{s}"] = f"{prefix}count_non_duplicate"
        rename_map[f"reverse_count{s}"] = f"{prefix}reverse_count"
        rename_map[f"forward_count{s}"] = f"{prefix}forward_count"
        rename_map[f"pass_alt_reads{s}"] = f"{prefix}pass_alt_reads"
        rename_map[f"scst_num_reads{s}"] = f"{prefix}scst_num_reads"
        rename_map[f"sced_num_reads{s}"] = f"{prefix}sced_num_reads"

        # ref/nonref columns from PILEUP
        for i in range(len(PILEUP_CONFIG.positions)):
            rename_map[f"ref{i}{s}"] = f"{prefix}ref{i}"
            rename_map[f"nonref{i}{s}"] = f"{prefix}nonref{i}"

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

    return variants_df


# =============================================================================
# STEP 3: Prepare Data + Run Classifier
# =============================================================================


def run_classifier(
    df_variants: pl.DataFrame,
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
    # TODO: don't convert to pandas, keep as polars dataframe once model compatibility verified
    df_variants_pandas = df_variants.to_pandas()

    xgb_clf = somatic_featuremap_inference_utils.load_xgb_model(xgb_model_path)
    model_features = xgb_clf.get_booster().feature_names
    logger.info(f"Loaded model with features: {model_features}")

    df_variants_pandas["xgb_proba"] = somatic_featuremap_inference_utils.predict(xgb_clf, df_variants_pandas)
    return df_variants_pandas


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


@app.default
def run(
    somatic_featuremap: Path,
    output_vcf: Path,
    genome_file: Path,
    ref_tr_file: Path,
    xgb_model_file: Path,
    *,
    filter_string: str = "PASS",
    write_agg_params: bool = False,
    verbose: bool = False,
) -> None:
    """Classify somatic featuremap variants using XGBoost model.

    Runs the full classification pipeline:
    1. Filter VCF and add TR annotations
    2. Convert VCF to dataframe with aggregations
    3. Run XGBoost classifier
    4. Write enhanced VCF with predictions

    Parameters
    ----------
    somatic_featuremap
        Somatic featuremap VCF file (gzipped, indexed).
    output_vcf
        Output VCF file path.
    genome_file
        Genome FASTA index (.fai) file.
    ref_tr_file
        Reference tandem repeat file in BED format.
    xgb_model_file
        XGBoost model file for inference.
    filter_string
        Filter tags to apply on the VCF file.
    write_agg_params
        Write aggregated parameters to output VCF.
    verbose
        Enable verbose output and debug messages.

    Examples
    --------
    $ somatic_featuremap_fields_transformation input.vcf.gz -o output.vcf.gz \\
        --genome-file genome.fa.fai --ref-tr-file tandem_repeats.bed \\
        --xgb-model-file model.json
    """
    # Ensure output has .vcf.gz suffix
    if not str(output_vcf).endswith(".vcf.gz"):
        logger.debug("adding .vcf.gz suffix to the output vcf file")
        output_vcf = output_vcf.with_suffix(".vcf.gz")

    # Use output directory for temporary files
    out_dir = output_vcf.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with console.status("[bold cyan]Running somatic featuremap classifier..."):
        somatic_featuremap_classifier(
            somatic_featuremap_vcf_path=somatic_featuremap,
            ref_tr_file=ref_tr_file,
            genome_file=genome_file,
            out_dir=out_dir,
            filter_string=filter_string,
            xgb_model_path=xgb_model_file,
            output_vcf=output_vcf,
            write_agg_params=write_agg_params,
            verbose=verbose,
        )

    console.print(f"[green]✓[/green] Output VCF written to: {output_vcf}")


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
    # Save parquet alongside output VCF
    output_parquet_path = output_vcf.with_suffix(".parquet")
    df_polars = read_vcf_with_aggregation(sfm_filtered_with_tr, output_parquet_path)

    if df_polars.is_empty():
        # TODO: handle better
        logger.warning(f"No variants found in the input VCF: {somatic_featuremap_vcf_path}")

    tumor_sample, normal_sample = get_sample_names_from_vcf(somatic_featuremap_vcf_path)
    samples = [tumor_sample, normal_sample]

    logger.info("Calculating PILEUP-based ref/nonref features")
    df_polars = calculate_pileup_features(df_polars, tumor_sample, normal_sample)
    logger.info("Post-processing the aggregated dataframe")
    df_variants = aggregated_df_post_processing(df_polars, samples)

    logger.info("Step 3: Running the classifier")
    df_variants = rename_cols_for_model(df_variants, samples)
    df_variants = run_classifier(df_variants, xgb_model_path)

    logger.info("Step 4: Writing enhanced VCF")
    write_enhanced_vcf(somatic_featuremap_vcf_path, output_vcf, df_variants, write_agg_params)
    logger.info(f"Output VCF written to: {output_vcf}")

    return output_vcf


def main():
    """Main entry point for the script."""
    app()


if __name__ == "__main__":
    main()
