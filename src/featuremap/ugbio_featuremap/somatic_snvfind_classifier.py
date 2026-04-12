import logging
import re
import sys
import tempfile
from pathlib import Path

import cyclopts
import polars as pl
import pysam
import xgboost
from ugbio_core.exec_utils import print_and_execute
from ugbio_core.logger import logger
from ugbio_core.misc_utils import cleanup_temp_files
from ugbio_core.vcf_utils import VcfMetaType, VcfUtils, get_vcf_sample_names, write_vcf_header_file

from ugbio_featuremap import somatic_snvfind_inference_utils
from ugbio_featuremap.featuremap_to_dataframe import vcf_to_parquet
from ugbio_featuremap.featuremap_utils import FeatureMapFields, FeatureMapFilters
from ugbio_featuremap.somatic_snvfind_utils import (
    DEFAULT_XGB_PROBA_THRESHOLD,
    GT_FORMAT_FIELD,
    NORMAL_PREFIX,
    PILEUP_CONFIG,
    REQUIRED_FORMAT_FIELDS,
    REQUIRED_INFO_FIELDS,
    TR_CONFIG,
    TUMOR_PREFIX,
    XGB_PROBA_INFO_FIELD,
    _log_regions_bed_preview,
    lowqual_filter_header_line,
)

app = cyclopts.App(
    help="Classify somatic snvfind variants using XGBoost model.",
)


# =============================================================================
# STEP 1: Filter for PASS + Add Tandem Repeat
# =============================================================================
def filter_and_annotate_tr(
    input_vcf: Path,
    tandem_repeats_bed: Path,
    genome_index_file: Path,
    out_dir: Path,
    filter_string: str | None = "PASS",
    regions_bed_file: Path | None = None,
    n_threads: int = 1,
) -> Path:
    """
    Filter VCF and annotate with tandem repeat features in a single pass.

    This unified preprocessing function:
    1. Filters the VCF to specified regions (if regions_bed_file provided)
    2. Filters the VCF to keep only specified variants (e.g., PASS)
    3. Annotates the filtered variants with tandem repeat information

    Parameters
    ----------
    input_vcf : Path
        Path to the input VCF file (gzipped).
    tandem_repeats_bed : Path
        Path to the reference tandem repeat BED file.
    genome_index_file : Path
        Path to the reference genome FASTA index file (.fai).
    out_dir : Path
        Output directory for the processed VCF file.
    filter_string : str, optional
        FILTER value to keep (e.g., "PASS"). If None, no filtering is applied.
        Defaults to "PASS".
    regions_bed_file : Path, optional
        BED file specifying regions to process. If provided, only variants
        within these regions are processed. Defaults to None (process all regions).
    n_threads : int, optional
        Number of threads for bcftools operations. Defaults to 1.

    Returns
    -------
    Path
        Path to the output VCF file with FILTER applied and TR annotations added.
        The output file will have '.filtered.tr_info.vcf.gz' suffix.
    """
    vcf_utils = VcfUtils()

    with tempfile.TemporaryDirectory(dir=out_dir) as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Build extra_args for bcftools view - combine region and filter options
        extra_args_parts = []
        if regions_bed_file:
            logger.info(f"Restricting to regions from: {regions_bed_file}")
            _log_regions_bed_preview(regions_bed_file)
            extra_args_parts.append(f"-R {regions_bed_file}")
        if filter_string:
            logger.info(f"Filtering VCF to keep variants with FILTER={filter_string}")
            extra_args_parts.append(f"-f {filter_string}")

        # Step 1: Filter VCF by regions and/or FILTER status
        if extra_args_parts:
            # Strip extension first, then append new suffix to avoid double .vcf.gz in the name
            filtered_vcf = tmpdir_path / (input_vcf.name.replace(".vcf.gz", "") + ".filtered.vcf.gz")
            extra_args = " ".join(extra_args_parts)
            vcf_utils.view_vcf(str(input_vcf), str(filtered_vcf), n_threads=n_threads, extra_args=extra_args)
            vcf_utils.index_vcf(str(filtered_vcf))
            vcf_to_annotate = filtered_vcf
        else:
            vcf_to_annotate = input_vcf

        # Step 2: Create TR annotation file (fully piped: bcftools -> bedtools -> cut -> sort -> bgzip)
        logger.info(f"Creating TR annotation file for {vcf_to_annotate}")
        gz_tsv, hdr_file = _create_tr_annotation_file(
            vcf_to_annotate, tandem_repeats_bed, genome_index_file, tmpdir_path
        )

        # Step 3: Annotate VCF with TR fields
        logger.info("Annotating VCF with tandem repeat information")
        output_suffix = ".filtered.tr_info.vcf.gz" if filter_string else ".tr_info.vcf.gz"
        output_vcf = out_dir / (input_vcf.name.replace(".vcf.gz", "") + output_suffix)
        vcf_utils.annotate_vcf(
            input_vcf=str(vcf_to_annotate),
            output_vcf=str(output_vcf),
            annotation_file=str(gz_tsv),
            header_file=str(hdr_file),
            columns=TR_CONFIG.get_bcftools_annotate_columns(),
            n_threads=n_threads,
        )

    logger.info(f"Filtered and TR-annotated VCF written to: {output_vcf}")
    return output_vcf


def _create_tr_annotation_file(
    input_vcf: Path, tandem_repeats_bed: Path, genome_index_file: Path, tmpdir: Path
) -> tuple[Path, Path]:
    """Create TR annotation file from VCF in one piped command.

    Pipeline: bcftools query -> bedtools closest -> cut -> sort -> bgzip
    Then: tabix for indexing
    """
    gz_tsv = tmpdir / "tr_annotation.tsv.gz"
    cmd = (
        f"bcftools query -f '%CHROM\\t%POS0\\t%END\\n' {input_vcf} | "
        f"bedtools closest -D ref -g {genome_index_file} -a stdin -b {tandem_repeats_bed} | "
        f"cut -f1,3,5-10 | "
        f"sort -k1,1 -k2,2n | "
        f"bgzip -c"
    )
    print_and_execute(cmd, output_file=str(gz_tsv), log_level=logging.DEBUG)

    cmd = f"tabix -s 1 -b 2 -e 2 {gz_tsv}"
    print_and_execute(cmd, log_level=logging.DEBUG)

    # Create header file for bcftools annotate
    hdr_file = tmpdir / "tr_hdr.txt"
    write_vcf_header_file(list(TR_CONFIG.info_fields), hdr_file)

    return gz_tsv, hdr_file


# =============================================================================
# STEP 2: Convert VCF to DataFrame + Post-Processing
# =============================================================================


def get_fields_to_drop_from_vcf(vcf_path: Path) -> tuple[set[str], set[str]]:
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


def read_vcf_with_aggregation(
    vcf_path: Path,
    output_parquet_path: Path,
    tumor_sample: str,
    normal_sample: str,
    n_threads: int = 1,
) -> pl.DataFrame:
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
    tumor_sample : str
        Tumor sample name.
    normal_sample : str
        Normal sample name.
    n_threads : int, optional
        Number of parallel jobs for VCF to Parquet conversion. Defaults to 1.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with aggregated features and sample-suffixed columns.
    """
    # Compute which columns to drop based on required fields
    drop_info, drop_format = get_fields_to_drop_from_vcf(vcf_path)

    vcf_to_parquet(
        vcf=str(vcf_path),
        out=str(output_parquet_path),
        drop_info=drop_info,
        drop_format=drop_format,
        jobs=n_threads,
        list_mode="aggregate",
        expand_columns={"AD": 2},  # Split AD into AD_0 (ref), AD_1 (alt)
    )

    logger.info(f"Read aggregated dataframe from parquet file: {output_parquet_path}")
    aggregated_df = pl.read_parquet(output_parquet_path)

    if aggregated_df.is_empty():
        logger.error(f"No data found in the aggregated dataframe: {output_parquet_path}")
        raise RuntimeError(f"No data found in the aggregated dataframe: {output_parquet_path}")

    logger.info(f"Loaded {len(aggregated_df):,} variants from parquet: {output_parquet_path}")

    logger.info("Calculating PILEUP-based ref/nonref features")
    aggregated_df = calculate_pileup_features(aggregated_df, tumor_sample, normal_sample)

    logger.info("Post-processing the aggregated dataframe")
    aggregated_df = aggregated_df_post_processing(aggregated_df, sample_names=[tumor_sample, normal_sample])

    return aggregated_df


def calculate_ref_nonref_columns(variants_df: pl.DataFrame, sample_suffix: str) -> pl.DataFrame:
    """
    Calculate REF and NON_REF columns from PILEUP columns for each position.

    For each position, compare to the appropriate reference:
    - L2 → X_PREV2
    - L1 → X_PREV1
    - C → REF
    - R1 → X_NEXT1
    - R2 → X_NEXT2

    For each position (L2, L1, C, R1, R2):
    - REF_{pos} = PILEUP_{reference_base}_{pos}_{sample}
    - NON_REF_{pos} = sum of PILEUP_{non-reference bases}_{pos} + PILEUP_{DEL}_{pos} + PILEUP_{INS}_{pos}

    Output columns follow sample suffix convention (e.g., REF_C_Pa_46_FreshFrozen).

    Parameters
    ----------
    variants_df : pl.DataFrame
        DataFrame with PILEUP columns.
    sample_suffix : str
        Sample suffix from vcf_to_parquet (e.g., "_Pa_46_FreshFrozen").

    Returns
    -------
    pl.DataFrame
        DataFrame with REF_{pos} and NON_REF_{pos} columns added (with sample suffix).
    """
    ref_nonref_exprs = []

    for pos in PILEUP_CONFIG.positions:
        ref_col = PILEUP_CONFIG.get_reference_column(pos)

        # Build ref column: select the PILEUP column matching the reference base for this position
        ref_expr = pl.when(pl.lit(value=False)).then(pl.lit(None))  # Start with a dummy condition that never matches
        for base in PILEUP_CONFIG.bases:
            ref_expr = ref_expr.when(pl.col(ref_col) == base).then(
                pl.col(PILEUP_CONFIG.get_column_name(base, pos, sample_suffix))
            )
        ref_expr = ref_expr.otherwise(pl.lit(0)).fill_null(0).alias(f"REF_{pos}{sample_suffix}")
        ref_nonref_exprs.append(ref_expr)

        # Build nonref column: sum of non-reference bases + DEL + INS
        nonref_components = []

        # Add non-reference base counts (exclude the reference base for this position)
        for base in PILEUP_CONFIG.bases:
            col_name = PILEUP_CONFIG.get_column_name(base, pos, sample_suffix)
            if col_name in variants_df.columns:
                nonref_components.append(
                    pl.when(pl.col(ref_col) != base).then(pl.col(col_name).fill_null(0)).otherwise(pl.lit(0))
                )

        # Add DEL and INS counts (always included in nonref)
        for indel in PILEUP_CONFIG.indels:
            col_name = PILEUP_CONFIG.get_column_name(indel, pos, sample_suffix)
            if col_name in variants_df.columns:
                nonref_components.append(pl.col(col_name).fill_null(0))

        if nonref_components:
            nonref_expr = pl.sum_horizontal(nonref_components).alias(f"NON_REF_{pos}{sample_suffix}")
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


def aggregated_df_post_processing(variants_df: pl.DataFrame, sample_names: list[str]) -> pl.DataFrame:
    """
    Post-process the aggregated Polars DataFrame to include all required features by ML inference.
    Derives additional columns from aggregation statistics (with sample suffix)
    E.g.  add Sum, count non zero and add ref/nonref columns for PILEUP columns, etc.


    Parameters
    ----------
    variants_df : pl.DataFrame
        DataFrame from vcf_to_parquet with aggregate mode.
    sample_names : list[str]
        List of sample names.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with derived columns for ML inference.
    """

    # Derive columns for each sample (using sample suffix convention)
    derived_exprs = []

    for sample_name in sample_names:
        s = f"_{sample_name}"  # Original suffix from vcf_to_parquet

        # Count of non-zero values = count - count_zero
        for field in (
            FeatureMapFields.DUP,
            FeatureMapFields.REV,
            FeatureMapFields.FILT,
            FeatureMapFields.SCST,
            FeatureMapFields.SCED,
        ):
            derived_exprs.append(
                (pl.col(f"{field.value}_count{s}") - pl.col(f"{field.value}_count_zero{s}"))
                .cast(pl.Int64)
                .alias(f"{field.value}_count_non_zero{s}")
            )

    # Apply derived columns
    variants_df = variants_df.with_columns(derived_exprs)

    return variants_df


# =============================================================================
# STEP 3: Prepare Data + Run Classifier
# =============================================================================


def _build_sample_rename_map(columns: list[str], sample_name: str, prefix: str) -> dict[str, str]:
    """Replace sample name suffix with the t_/n_ prefix for all matching columns.

    For any column ending with ``_{sample_name}``, produce a mapping
    ``original_col -> {prefix}{base}`` where *base* is the column name
    without the sample suffix.
    e.g. MQUAL_mean_Pa46 → t_MQUAL_mean

    Parameters
    ----------
    columns : list[str]
        Column names to scan.
    sample_name : str
        Sample name to look for as a suffix.
    prefix : str
        Prefix to prepend (e.g. ``"t_"`` or ``"n_"``).

    Returns
    -------
    dict[str, str]
        Mapping from original column name to prefixed name.
    """
    suffix = f"_{sample_name}"
    return {col: f"{prefix}{col[: -len(suffix)]}" for col in columns if col.endswith(suffix)}


def rename_cols_for_model(variants_df: pl.DataFrame, samples: list[str], vcf_path: Path | None = None) -> pl.DataFrame:
    """
    Rename the dataframe columns to match the model expected features.

    Parameters
    ----------
    variants_df : pl.DataFrame
        DataFrame with sample-suffixed columns.
    samples : list[str]
        List of sample names (tumor first, normal second).
    vcf_path : Path, optional
        Path to the VCF file. If provided, the sample names will be validated against the VCF file.
        This is used for debugging purposes.

    Returns
    -------
    pl.DataFrame
        DataFrame with renamed columns.
    """
    logger.debug("Renaming columns for model inference")
    logger.debug("Validating sample names against VCF file")
    if vcf_path:
        samples_from_vcf = get_vcf_sample_names(vcf_path)
        if len(samples_from_vcf) != 2:  # noqa: PLR2004
            raise ValueError(f"Expected exactly 2 samples in VCF, found {len(samples_from_vcf)}: {samples_from_vcf}")
        tumor_sample, normal_sample = samples_from_vcf[0], samples_from_vcf[1]
        if tumor_sample not in samples or normal_sample not in samples:
            raise ValueError(
                f"The given samples ({samples}) do not match the sample names in the VCF file: "
                f"{tumor_sample}, {normal_sample}"
            )

    logger.debug("Building rename map")
    rename_map = {}

    # Non-sample columns (no sample suffix)
    rename_map[FeatureMapFields.CHROM.value] = f"{TUMOR_PREFIX}chrom"
    rename_map[FeatureMapFields.POS.value] = f"{TUMOR_PREFIX}pos"
    rename_map[FeatureMapFields.REF.value] = "ref_allele"
    rename_map[FeatureMapFields.ALT.value] = "alt_allele"

    # INFO fields that should NOT get tumor prefix (variant-level, not sample-specific)
    variant_level_info_fields = {
        FeatureMapFields.X_HMER_REF.value,
        FeatureMapFields.X_HMER_ALT.value,
    }

    # INFO fields with tumor prefix (sample-context INFO fields like TR)
    for info_field in REQUIRED_INFO_FIELDS:
        if info_field in variant_level_info_fields:
            # Keep variant-level fields without prefix (will be lowercased)
            rename_map[info_field] = info_field
        else:
            rename_map[info_field] = f"{TUMOR_PREFIX}{info_field}"

    # Sample specific columns
    sample_prefix_tuples = [(samples[0], TUMOR_PREFIX), (samples[1], NORMAL_PREFIX)]

    # Generic: replace _{sample_name} suffix with t_/n_ prefix for all sample-suffixed columns
    for sample_name, prefix in sample_prefix_tuples:
        rename_map.update(_build_sample_rename_map(variants_df.columns, sample_name, prefix))

    # Override: model-specific names that don't follow the generic pattern
    for sample_name, prefix in sample_prefix_tuples:
        s = f"_{sample_name}"

        # Count zero for MAPQ (mapq0_count)
        rename_map[f"{FeatureMapFields.MAPQ.value}_count_zero{s}"] = f"{prefix}map0_count"

        # alt_reads from AD_1
        rename_map[f"AD_1{s}"] = f"{prefix}alt_reads"

        # 0/1 fields
        rename_map[f"{FeatureMapFields.DUP.value}_count_non_zero{s}"] = f"{prefix}count_duplicate"
        rename_map[f"{FeatureMapFields.DUP.value}_count_zero{s}"] = f"{prefix}count_non_duplicate"
        rename_map[f"{FeatureMapFields.REV.value}_count_non_zero{s}"] = f"{prefix}reverse_count"
        rename_map[f"{FeatureMapFields.REV.value}_count_zero{s}"] = f"{prefix}forward_count"
        rename_map[f"{FeatureMapFields.FILT.value}_count_non_zero{s}"] = f"{prefix}pass_alt_reads"
        rename_map[f"{FeatureMapFields.SCST.value}_count_non_zero{s}"] = f"{prefix}scst_num_reads"
        rename_map[f"{FeatureMapFields.SCED.value}_count_non_zero{s}"] = f"{prefix}sced_num_reads"

        # ref/nonref columns from PILEUP (map new names back to model-expected names)
        for i, pos in enumerate(PILEUP_CONFIG.positions):
            rename_map[f"REF_{pos}{s}"] = f"{prefix}ref{i}"
            rename_map[f"NON_REF_{pos}{s}"] = f"{prefix}nonref{i}"

    # Lowercase all target names
    rename_map = {k: v.lower() for k, v in rename_map.items()}

    missing_cols = set(rename_map.keys()) - set(variants_df.columns)
    if missing_cols:
        logger.error(f"Columns intended for renaming are missing from the dataframe: {sorted(missing_cols)}")
        raise ValueError(f"Columns intended for renaming are missing from the dataframe: {sorted(missing_cols)}")

    variants_df = variants_df.rename(rename_map)

    return variants_df


def run_classifier(
    df_variants: pl.DataFrame,
    xgb_model_path: Path,
) -> pl.Series:
    """
    Run XGBoost classifier on the prepared DataFrame.

    Parameters
    ----------
    df_variants : pl.DataFrame
        DataFrame with features prepared for ML inference
    xgb_model_path : Path
        Path to the XGBoost model file.

    Returns
    -------
    pl.Series
        Series with xgb_proba predictions.
    """
    # TODO: don't convert to pandas, keep as polars dataframe once model compatibility verified
    df_variants_pandas = df_variants.to_pandas()

    xgb_clf = xgboost.XGBClassifier()
    xgb_clf.load_model(xgb_model_path)
    model_features = xgb_clf.get_booster().feature_names
    logger.debug(f"Model features: {model_features}")

    predictions = somatic_snvfind_inference_utils.predict(xgb_clf, df_variants_pandas)
    logger.info(f"Classified {len(predictions):,} variants with XGBoost model")

    return pl.Series(FeatureMapFields.XGB_PROBA.value.lower(), predictions)


# =============================================================================
# STEP 4: Annotate VCF with XGBoost probability
# =============================================================================


def _get_classification_bcftools_columns() -> str:
    """Return the columns string for bcftools annotate with XGB_PROBA, FILTER, and GT.

    When annotating from a VCF source, only the fields to transfer are specified
    (CHROM/POS/REF/ALT matching is implicit).
    """
    return (
        f"{VcfMetaType.INFO.value}/{FeatureMapFields.XGB_PROBA.value},"
        f"{FeatureMapFields.FILTER.value},"
        f"{VcfMetaType.FORMAT.value}/{FeatureMapFields.GT.value}"
    )


def annotate_and_classify_vcf(
    input_vcf_path: Path,
    output_vcf_path: Path,
    df_variants: pl.DataFrame,
    tumor_sample: str,
    normal_sample: str,
    xgb_proba_threshold: float = DEFAULT_XGB_PROBA_THRESHOLD,
    n_threads: int = 1,
) -> None:
    """
    Annotate VCF with XGBoost probability, FILTER assignment, and GT using bcftools annotate.

    Creates a VCF annotation file (bcftools requires GT from a VCF source) with
    XGB_PROBA, FILTER, and per-sample GT, then uses bcftools annotate with `-x`
    to remove old FILTER definitions.

    PASS records get FILTER=PASS, tumor GT=0/1, normal GT=./. (missing).
    Non-PASS records get FILTER=LowQual, tumor GT=./. (missing), normal GT=./. (missing).

    Parameters
    ----------
    input_vcf_path : Path
        Path to the input VCF file.
    output_vcf_path : Path
        Path to the output VCF file.
    df_variants : pl.DataFrame
        Polars DataFrame containing variant information with xgb_proba column.
        Must have columns: CHROM, POS, REF, ALT, xgb_proba
    tumor_sample : str
        Tumor sample name (for per-sample GT annotation and header line).
    normal_sample : str
        Normal sample name (for per-sample GT annotation).
    xgb_proba_threshold : float, optional
        Threshold for PASS/LowQual FILTER assignment. Defaults to DEFAULT_XGB_PROBA_THRESHOLD.
    n_threads : int, optional
        Number of threads for bcftools. Defaults to 1.
    """
    work_dir = output_vcf_path.parent
    annotation_vcf = work_dir / "classification_annotations.vcf"
    annotation_vcf_gz = work_dir / "classification_annotations.vcf.gz"
    header_file = work_dir / "classification_header.txt"

    xgb_col = FeatureMapFields.XGB_PROBA.value.lower()

    # Round xgb_proba first so that FILTER, GT, and the displayed INFO value all use the same number
    xgb_rounded = pl.col(xgb_col).round(3)

    # Build annotation DataFrame in VCF format (bcftools requires GT from a VCF source, not TSV)
    annotation_df = df_variants.select(
        [
            pl.col(FeatureMapFields.CHROM.value).alias("#CHROM"),
            pl.col(FeatureMapFields.POS.value).alias("POS"),
            pl.lit(".").alias("ID"),
            pl.col(FeatureMapFields.REF.value).alias("REF"),
            pl.col(FeatureMapFields.ALT.value).alias("ALT"),
            pl.lit(".").alias("QUAL"),
            pl.when(xgb_rounded >= xgb_proba_threshold)
            .then(pl.lit(FeatureMapFilters.PASS.value))
            .otherwise(pl.lit(FeatureMapFilters.LOW_QUAL.value))
            .alias("FILTER"),
            (pl.lit(f"{FeatureMapFields.XGB_PROBA.value}=") + xgb_rounded.cast(pl.Utf8)).alias(VcfMetaType.INFO.value),
            pl.lit(FeatureMapFields.GT.value).alias(VcfMetaType.FORMAT.value),
            pl.when(xgb_rounded >= xgb_proba_threshold)
            .then(pl.lit("0/1"))
            .otherwise(pl.lit("./."))
            .alias(tumor_sample),
            pl.lit("./.").alias(normal_sample),
        ]
    ).sort(["#CHROM", "POS"])

    # Write annotation VCF: meta-header lines + column header + data rows
    vcf_meta = (
        "##fileformat=VCFv4.2\n"
        f"{XGB_PROBA_INFO_FIELD.to_header_line()}\n"
        f"{lowqual_filter_header_line(xgb_proba_threshold)}\n"
        f"{GT_FORMAT_FIELD.to_header_line()}\n"
    )
    with open(annotation_vcf, "w") as f:
        f.write(vcf_meta)
        annotation_df.write_csv(f, separator="\t", include_header=True, quote_style="never")
    logger.debug(f"Created annotation VCF with {len(annotation_df):,} records: {annotation_vcf}")

    # Compress with bgzip and index with tabix
    print_and_execute(f"bgzip -f {annotation_vcf}", log_level=logging.DEBUG)
    logger.debug(f"Compressed annotation file: {annotation_vcf_gz}")

    print_and_execute(f"tabix -p vcf {annotation_vcf_gz}", log_level=logging.DEBUG)
    logger.debug(f"Indexed annotation file: {annotation_vcf_gz}.tbi")

    # Read input VCF header to determine which FILTER definitions to remove and whether GT exists
    with pysam.VariantFile(str(input_vcf_path)) as vcf:
        filters_to_remove = [f for f in vcf.header.filters if f != FeatureMapFilters.PASS.value]
        has_gt = FeatureMapFields.GT.value in vcf.header.formats

    # Create header file with additional lines (tumor_sample, command, LowQual FILTER, GT FORMAT)
    additional_header_lines = [f"##tumor_sample={tumor_sample}"]
    try:
        command = " ".join(sys.argv[1:])
        additional_header_lines.append(f"##somatic_snvfind_classifier={command}")
    except Exception:
        logger.warning("Could not get command line arguments to add to header. skipping.")

    additional_header_lines.append(lowqual_filter_header_line(xgb_proba_threshold))
    if not has_gt:
        additional_header_lines.append(GT_FORMAT_FIELD.to_header_line())

    write_vcf_header_file([XGB_PROBA_INFO_FIELD], header_file, additional_header_lines=additional_header_lines)
    logger.debug(f"Created header file: {header_file}")

    # Build extra_args: -x to remove old filter definitions
    extra_args = ""
    if filters_to_remove:
        remove_str = ",".join(f"FILTER/{f}" for f in filters_to_remove)
        extra_args = f"-x {remove_str}"

    # Annotate VCF using bcftools annotate (VCF annotation source for GT support)
    vcf_utils = VcfUtils()
    vcf_utils.annotate_vcf(
        input_vcf=str(input_vcf_path),
        output_vcf=str(output_vcf_path),
        annotation_file=str(annotation_vcf_gz),
        header_file=str(header_file),
        columns=_get_classification_bcftools_columns(),
        n_threads=n_threads,
        extra_args=extra_args,
    )
    logger.info(
        f"Annotated VCF with {FeatureMapFields.XGB_PROBA.value}, "
        f"{FeatureMapFields.FILTER.value}, and {FeatureMapFields.GT.value}: {output_vcf_path}"
    )

    # Cleanup temporary files
    cleanup_temp_files([annotation_vcf_gz, Path(f"{annotation_vcf_gz}.tbi"), header_file])


# =============================================================================
# MAIN ORCHESTRATION FUNCTION
# =============================================================================
def validate_inputs_and_prepare_output(
    output_vcf: Path,
    filter_string: str,
    *,
    somatic_snvfind_vcf: Path,
    genome_index_file: Path,
    tandem_repeats_bed: Path,
    xgb_model_json: Path,
    regions_bed_file: Path | None = None,
    verbose: bool = False,
) -> tuple[Path, Path]:
    """Validate input files and prepare the output directory. Also sets up logging if verbose is True.

    Checks that all required input files exist, validates the filter_string format,
    ensures the output VCF path has a .vcf.gz suffix, and creates the output directory.

    Returns
    -------
    tuple[Path, Path]
        (output_vcf, out_dir) - the resolved output VCF path and the output directory.

    Raises
    ------
    FileNotFoundError
        If any required input file does not exist.
    ValueError
        If filter_string contains invalid characters.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    if filter_string and not re.fullmatch(r"[a-zA-Z0-9_-]+", filter_string):
        raise ValueError(
            f"filter_string must contain only alphanumeric characters, underscores, and hyphens; got: {filter_string!r}"
        )

    input_files = [somatic_snvfind_vcf, genome_index_file, tandem_repeats_bed, xgb_model_json]
    if regions_bed_file is not None:
        input_files.append(regions_bed_file)
    if not all(path.exists() for path in input_files):
        raise FileNotFoundError(
            f"Input file does not exist: {[path.name for path in input_files if not path.exists()]}"
        )

    # Ensure output has .vcf.gz suffix
    if not output_vcf.name.endswith(".vcf.gz"):
        if output_vcf.name.endswith(".vcf"):
            output_vcf = output_vcf.parent / (output_vcf.name + ".gz")
        else:
            output_vcf = output_vcf.with_suffix(".vcf.gz")
        logger.info(f"Adjusted output file to: {output_vcf.name}")

    # create output directory
    out_dir = output_vcf.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return output_vcf, out_dir


@app.default
def somatic_snvfind_classifier(  # noqa: PLR0913
    somatic_snvfind_vcf: Path,
    output_vcf: Path,
    genome_index_file: Path,
    tandem_repeats_bed: Path,
    xgb_model_json: Path,
    *,
    output_parquet: Path | None = None,
    filter_string: str = "PASS",
    regions_bed_file: Path | None = None,
    xgb_proba_thresh: float = DEFAULT_XGB_PROBA_THRESHOLD,
    n_threads: int = 1,
    verbose: bool = False,
) -> tuple[Path, Path]:
    """
    Classify somatic snvfind variants using XGBoost model.

    Steps:
    1. Filter VCF and add TR annotations
    2. Convert VCF to (polars) dataframe and add transformations such as aggregations and PILEUP-based
     ref/nonref features
    3. Run XGBoost classifier
    4. Annotate VCF with XGBoost probability using bcftools annotate

    Parameters
    ----------
    somatic_snvfind_vcf
        Somatic snvfind VCF file (gzipped, indexed).
    output_vcf
        Output VCF file path.
    genome_index_file
        Genome FASTA index (.fai) file.
    tandem_repeats_bed
        Reference tandem repeat file in BED format (sorted by chromosome and start position).
    xgb_model_json
        XGBoost model file for inference.
    output_parquet
        Output Parquet file path. If not provided, a default path will be used.
    filter_string
        Filter tags to apply on the VCF file.
    regions_bed_file
        Optional BED file specifying regions to process. When provided, only
        variants within these regions are processed. Useful for quick testing
        or targeted analysis on specific genomic regions.
    xgb_proba_thresh
        Threshold for PASS/LowQual FILTER assignment based on xgb_proba.
    n_threads
        Number of threads for parallel processing.
    verbose
        Enable verbose output and debug messages.

    Examples
    --------
    $ somatic_snvfind_classifier input.vcf.gz -o output.vcf.gz \\
        --genome-file genome.fa.fai --tandem-repeats-bed tandem_repeats.bed \\
        --xgb-model-json model.json

    $ somatic_snvfind_classifier input.vcf.gz -o output.vcf.gz \\
        --genome-file genome.fa.fai --tandem-repeats-bed tandem_repeats.bed \\
        --xgb-model-json model.json --regions-bed-file chr1_test.bed
    """
    output_vcf, out_dir = validate_inputs_and_prepare_output(
        output_vcf,
        filter_string,
        somatic_snvfind_vcf=somatic_snvfind_vcf,
        genome_index_file=genome_index_file,
        tandem_repeats_bed=tandem_repeats_bed,
        xgb_model_json=xgb_model_json,
        regions_bed_file=regions_bed_file,
        verbose=verbose,
    )

    samples = get_vcf_sample_names(somatic_snvfind_vcf)
    if len(samples) != 2:  # noqa: PLR2004
        raise ValueError(f"Expected exactly 2 samples in VCF, found {len(samples)}: {samples}")
    tumor_sample, normal_sample = samples[0], samples[1]

    logger.info(f"Processing samples: tumor={tumor_sample}, normal={normal_sample}")

    logger.info("Step 1: Filtering VCF and adding TR annotations")
    snvfind_filtered_with_tr = filter_and_annotate_tr(
        somatic_snvfind_vcf,
        tandem_repeats_bed,
        genome_index_file,
        out_dir,
        filter_string=filter_string,
        regions_bed_file=regions_bed_file,
        n_threads=n_threads,
    )

    logger.info("Step 2: Converting VCF to dataframe and adding transformations")
    if output_parquet is None:
        output_parquet = output_vcf.with_name(output_vcf.name.replace(".vcf.gz", "_snvfind.parquet"))
    logger.info(f"Output Parquet file: {output_parquet}")
    aggregated_df = read_vcf_with_aggregation(
        snvfind_filtered_with_tr, output_parquet, tumor_sample, normal_sample, n_threads=n_threads
    )

    logger.info("Step 3: Running the classifier")
    # Create a renamed copy for model inference
    samples = [tumor_sample, normal_sample]
    df_for_model = rename_cols_for_model(aggregated_df, samples)
    xgb_proba: pl.Series = run_classifier(df_for_model, xgb_model_json)

    # Add predictions to the aggregated dataframe
    aggregated_df = aggregated_df.with_columns(xgb_proba)

    # In debug mode, save the full processed DataFrame to parquet
    if verbose:
        aggregated_df.write_parquet(output_parquet)
        logger.debug(f"Saved debug parquet with all transformations and xgb_score: {output_parquet}")

    logger.info("Step 4: Annotating VCF with XGBoost probability, FILTER, and GT")
    annotate_and_classify_vcf(
        snvfind_filtered_with_tr,
        output_vcf,
        aggregated_df,
        tumor_sample=tumor_sample,
        normal_sample=normal_sample,
        xgb_proba_threshold=xgb_proba_thresh,
        n_threads=n_threads,
    )
    logger.info(f"Output VCF written to: {output_vcf}")

    # Clean up intermediate files
    cleanup_temp_files([snvfind_filtered_with_tr])

    return output_vcf, output_parquet


def main():
    """Main entry point for the script."""
    app()


if __name__ == "__main__":
    main()
