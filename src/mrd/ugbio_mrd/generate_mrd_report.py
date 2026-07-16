import argparse
import json
import os
import sys
from dataclasses import dataclass
from os.path import join as pjoin
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from ugbio_core.consts import FileExtension
from ugbio_core.logger import logger

import ugbio_mrd.mrd_utils as mrd
from ugbio_mrd.mrd_detection import DEFAULT_ALPHA, DEFAULT_LOD_FPR, run_detection_analysis
from ugbio_mrd.mrd_report_renderer import render_analysis_report, render_qc_report

RESULTS_HTML_REPORT = ".mrd_analysis_report.html"
QC_HTML_REPORT = ".mrd_qc_report.html"
BASE_PATH = Path(__file__).parent  # should be: src/mrd/ugbio_mrd

# Default thresholds for optional locus filters.
# Both filters are enabled by default when using the CLI.
# Pass None explicitly via MrdReportInputs to disable programmatically.
DEFAULT_THRESH_NOISE_LQ_READS: float = 0.7
DEFAULT_THRESH_MULTI_READ_PVALUE: float = 0.001
DEFAULT_READ_FILTER_QUERY: str = "filt>0 and snvq>60 and mapq>=60"
DEFAULT_LOD_RECALL: float = 0.95


@dataclass
class MrdReportInputs:
    """Inputs to generate a MRD report"""

    intersected_featuremaps_parquet: list[str]
    matched_signatures_vcf_files: list[str]
    control_signatures_vcf_files: list[str]
    coverage_bed: str
    output_dir: str
    output_basename: str
    featuremap_file: str
    srsnv_metadata_json: str
    db_control_signatures_vcf_files: list[str] = None
    tumor_sample: str = None
    signature_filter_query: str = None
    read_filter_query: str = None
    alpha: float = DEFAULT_ALPHA
    lod_fpr: float = DEFAULT_LOD_FPR
    lod_recall: float = DEFAULT_LOD_RECALL
    thresh_noise_lq_reads: float | None = DEFAULT_THRESH_NOISE_LQ_READS
    thresh_multi_read_pvalue: float | None = DEFAULT_THRESH_MULTI_READ_PVALUE


def generate_mrd_report(mrd_report_inputs: MrdReportInputs) -> tuple[Path, Path]:  # noqa: PLR0915, PLR0912, C901
    """
    Generate both the MRD analysis report (Jinja2 HTML) and the MRD QC report (Jinja2 HTML).

    Returns
    -------
    tuple[Path, Path]
        Paths to the analysis HTML report and the QC HTML report.
    """
    signatures_path, intersection_path = prepare_data_from_mrd_pipeline(mrd_report_inputs)

    results_html_path = Path(mrd_report_inputs.output_dir) / (mrd_report_inputs.output_basename + RESULTS_HTML_REPORT)
    qc_html_path = Path(mrd_report_inputs.output_dir) / (mrd_report_inputs.output_basename + QC_HTML_REPORT)

    # ── Analysis report: compute in Python, render via Jinja2 ──
    logger.info(f"Generating MRD analysis report (Jinja2). {results_html_path=}")

    signature_filter_query = (
        mrd_report_inputs.signature_filter_query or "(norm_coverage <= 2.5) and (norm_coverage >= 0.6)"
    )
    read_filter_query = mrd_report_inputs.read_filter_query or DEFAULT_READ_FILTER_QUERY

    # Normalise optional filter thresholds: 1.0 (noise) and None are both "disabled".
    # Centralise here so applied_filters and QC section both see the canonical value.
    thresh_noise_lq_reads = mrd_report_inputs.thresh_noise_lq_reads
    if thresh_noise_lq_reads is not None and thresh_noise_lq_reads >= 1.0:
        thresh_noise_lq_reads = None

    # 1. Load data
    df_features, df_features_filt, filtering_ratio, noise_excluded_loci = mrd.read_and_filter_features_parquet(
        intersection_path,
        read_filter_query,
        thresh_noise_lq_reads=thresh_noise_lq_reads,
    )
    df_signatures, df_signatures_filt = mrd.read_and_filter_signatures_parquet(
        signatures_path, signature_filter_query, filtering_ratio
    )
    denom_ratio, filt_ratio, _ = mrd.calc_tumor_fraction_denominator_ratio(
        mrd_report_inputs.featuremap_file, mrd_report_inputs.srsnv_metadata_json, read_filter_query
    )
    # Track all excluded loci (noise + multi-read) for coverage adjustment
    excluded_loci = noise_excluded_loci

    # 1.5. Multi-read locus filter: remove matched loci with unexpectedly many HQ reads
    # (e.g. germline / mosaic variants). Calibrated to the TF estimate from the current
    # df_features_filt; the pre-filter detection is saved for QC comparison.
    detection_pre_multi_read = None
    df_tf_pre_multi_read = None
    thresh_multi_read_pvalue = mrd_report_inputs.thresh_multi_read_pvalue
    if thresh_multi_read_pvalue is not None:
        df_tf_pre_multi_read, df_supporting_pre_multi = mrd.get_tf_from_filtered_data(
            df_features_filt,
            df_signatures_filt,
            plot_results=False,
            title="Filtered reads (before multi-read filter)",
            denom_ratio=denom_ratio,
            excluded_loci=excluded_loci,
        )
        detection_pre_multi_read = run_detection_analysis(
            df_tf=df_tf_pre_multi_read,
            df_signatures_filt=df_signatures_filt,
            alpha=mrd_report_inputs.alpha,
            lod_fpr=mrd_report_inputs.lod_fpr,
            lod_recall=mrd_report_inputs.lod_recall,
            df_supporting_reads_per_locus=df_supporting_pre_multi,
        )
        df_features_before_multi = df_features_filt
        df_features_filt, multi_read_info = mrd.apply_multi_read_locus_filter(
            df_features_filt,
            df_tf_pre_multi_read,
            df_signatures_filt,
            thresh_multi_read_pvalue,
        )
        # Collect loci removed by multi-read filter
        multi_read_excluded = df_features_before_multi.index.unique().difference(df_features_filt.index.unique())
        if excluded_loci is not None and len(excluded_loci) > 0:
            excluded_loci = excluded_loci.append(multi_read_excluded).unique()
        else:
            excluded_loci = multi_read_excluded

    df_tf_filt, df_supporting_reads_per_locus_filt = mrd.get_tf_from_filtered_data(
        df_features_filt,
        df_signatures_filt,
        plot_results=False,
        title="Filtered reads and signatures",
        denom_ratio=denom_ratio,
        excluded_loci=excluded_loci,
    )

    # 2. Run detection
    # When the multi-read filter is active the outlier loci have already been removed,
    # so the per-locus QC checks would trivially pass and add no information.
    # Pass None so run_detection_analysis skips those checks entirely.
    detection_per_locus = None if thresh_multi_read_pvalue is not None else df_supporting_reads_per_locus_filt
    detection = run_detection_analysis(
        df_tf=df_tf_filt,
        df_signatures_filt=df_signatures_filt,
        alpha=mrd_report_inputs.alpha,
        lod_fpr=mrd_report_inputs.lod_fpr,
        lod_recall=mrd_report_inputs.lod_recall,
        df_supporting_reads_per_locus=detection_per_locus,
    )

    # 3. Build applied filters
    filter_descriptions = {
        "ug_hcr": "In UG High Confidence Region",
        "giab_hcr": "In GIAB (HG001-007) High Confidence Region",
        "ug_mrd_blacklist": "Not in UG MRD Blacklist",
    }
    all_cols = list(df_signatures_filt.reset_index().columns) + list(df_signatures_filt.columns)
    applied_filters = {k: v for k, v in filter_descriptions.items() if k in all_cols}
    applied_filters["norm_coverage"] = signature_filter_query
    applied_filters["Read filter"] = read_filter_query
    if thresh_noise_lq_reads is not None:
        applied_filters["Noisy loci"] = f"lq_fraction > {thresh_noise_lq_reads}"
    if thresh_multi_read_pvalue is not None:
        _dataset_keys = [
            ("matched", "max_reads_per_locus"),
            ("control", "max_reads_per_locus_control"),
            ("db_control", "max_reads_per_locus_db_control"),
        ]
        _max_per_locus = ", ".join(f"{ds}={multi_read_info.get(key, 1)}" for ds, key in _dataset_keys)
        applied_filters["Multi-read locus filter"] = (
            f"Poisson outlier test; Bonferroni-corrected p < {thresh_multi_read_pvalue}; "
            f"max reads/locus: {_max_per_locus}"
        )

    # 4. SBS plot helper
    _sbs_colors = ["#1EBFF0", "#050708", "#E62725", "#CBCACB", "#A1C935", "#ECC6C5"]

    def plot_sbs_profile(df_sig, title="", ax=None, query=None):
        df_plot = df_sig if query is None else df_sig.query(query)
        _all_muts = ["C->A", "C->G", "C->T", "T->A", "T->C", "T->G"]
        counts = df_plot["mutation_type"].value_counts(normalize=True).reindex(_all_muts, fill_value=0)
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 2.4))
        bars = ax.barh(range(6), counts.to_numpy()[::-1], color=list(reversed(_sbs_colors)), height=0.65, linewidth=0)
        ax.set_yticks(range(6))
        ax.set_yticklabels([m.replace("->", ">") for m in reversed(_all_muts)], fontsize=9, fontweight="bold")
        for _j, (bar, val) in enumerate(zip(bars, counts.to_numpy()[::-1], strict=False)):
            ax.text(
                val + 0.003, bar.get_y() + bar.get_height() / 2, f"{val:.1%}", va="center", fontsize=8, color="#555"
            )
        ax.set_xlim(0, max(counts.to_numpy()) * 1.25 + 0.02)
        ax.set_xlabel("Fraction", fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.text(
            0.99,
            0.02,
            f"n = {len(df_plot):,}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#7f8c8d",
        )
        ax.spines[["top", "right"]].set_visible(False)
        return ax

    # 5. Render HTML
    def _fmt_files(files):
        if not files:
            return "—"
        return ", ".join(Path(f).name for f in files)

    try:
        import importlib.metadata
        import os
        import subprocess

        _pkg_version = importlib.metadata.version("ugbio-mrd")
        # Priority: GIT_COMMIT env var → git rev-parse (dev) → UGBIO_IMAGE_TAG
        # (baked into Docker image at build time) → plain package version.
        _commit = os.environ.get("GIT_COMMIT", "")
        if not _commit:
            try:
                _commit = subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            except Exception:  # noqa: BLE001
                _commit = ""
        if _commit:
            _version_str = f"{_pkg_version} ({_commit})"
        else:
            _image_tag = os.environ.get("UGBIO_IMAGE_TAG", "")
            _version_str = f"{_pkg_version} [{_image_tag}]" if _image_tag and _image_tag != "unknown" else _pkg_version
    except Exception:  # noqa: BLE001
        _version_str = "unknown"

    inputs_info = {
        "Sample": mrd_report_inputs.output_basename or "N/A",
        "ugbio-mrd version": _version_str,
        "Patient signature VCFs": _fmt_files(mrd_report_inputs.matched_signatures_vcf_files),
        "Control signature VCFs": _fmt_files(mrd_report_inputs.control_signatures_vcf_files),
        "Featuremap file": Path(mrd_report_inputs.featuremap_file).name if mrd_report_inputs.featuremap_file else "—",
        "Coverage BED": Path(mrd_report_inputs.coverage_bed).name if mrd_report_inputs.coverage_bed else "—",
        "Signature filter": signature_filter_query,
        "Read filter": read_filter_query,
    }

    html = render_analysis_report(
        detection=detection,
        df_tf=df_tf_filt,
        df_signatures=df_signatures,
        df_signatures_filt=df_signatures_filt,
        df_supporting_reads_per_locus=df_supporting_reads_per_locus_filt,
        basename=mrd_report_inputs.output_basename,
        signature_filter_query=signature_filter_query,
        read_filter_query=read_filter_query,
        denom_ratio=denom_ratio,
        filt_ratio=filt_ratio,
        plot_sbs_fn=plot_sbs_profile,
        plot_af_fn=mrd.plot_signature_allele_fractions,
        applied_filters=applied_filters,
        df_features=df_features,
        df_features_filt=df_features_filt,
        inputs_info=inputs_info,
    )
    results_html_path.write_text(html)

    # 6. Save detection JSON
    detection_json = {
        "call": detection.call,
        "detected": detection.detected,
        "p_value": detection.p_value,
        "matched_supporting_reads": detection.matched_supporting_reads,
        "matched_ctdna_vaf": detection.matched_ctdna_vaf,
        "null_median_reads": detection.null_median_reads,
        "null_max_reads": detection.null_max_reads,
        "n_synthetic_controls": detection.n_synthetic_controls,
        "detection_threshold": (
            detection.detection_threshold / detection.corrected_coverage
            if detection.detection_threshold is not None and detection.corrected_coverage > 0
            else None
        ),
        "personal_lod": detection.personal_lod,
        "signature_size": detection.signature_size,
        "mean_coverage": detection.mean_coverage,
        "corrected_coverage": detection.corrected_coverage,
        "qc_checks": [
            {"label": c.label, "value": c.value_str, "threshold": c.threshold_str, "passed": c.passed}
            for c in detection.qc_checks
        ],
        "alpha": detection.alpha,
    }
    detection_json_path = (
        Path(mrd_report_inputs.output_dir) / f"{mrd_report_inputs.output_basename}.detection_result.json"
    )
    with open(detection_json_path, "w") as f:
        json.dump(detection_json, f, indent=2, default=str)

    # 7. Save HDF5 tables (primary)
    output_h5_file = str(Path(mrd_report_inputs.output_dir) / f"{mrd_report_inputs.output_basename}.ctdna_vaf.h5")
    df_tf_filt.to_hdf(output_h5_file, key="df_ctdna_vaf_filt_signature_filt_featuremap", mode="w")
    df_supporting_reads_per_locus_filt.to_hdf(
        output_h5_file, key="df_supporting_reads_per_locus_filt_signature_filt_featuremap", mode="a"
    )

    # Save detection result as a single-row DataFrame
    detection_record = {
        "call": detection.call,
        "detected": detection.detected,
        "p_value": detection.p_value,
        "matched_supporting_reads": detection.matched_supporting_reads,
        "matched_ctdna_vaf": detection.matched_ctdna_vaf,
        "null_median_reads": detection.null_median_reads,
        "null_max_reads": detection.null_max_reads,
        "n_synthetic_controls": detection.n_synthetic_controls,
        "detection_threshold": (
            detection.detection_threshold / detection.corrected_coverage
            if detection.detection_threshold is not None and detection.corrected_coverage > 0
            else None
        ),
        "personal_lod": detection.personal_lod,
        "signature_size": detection.signature_size,
        "mean_coverage": detection.mean_coverage,
        "corrected_coverage": detection.corrected_coverage,
        "noise_rate": detection.noise_rate,
        "n_effective": detection.n_effective,
        "jeffreys_prior_applied": detection.jeffreys_prior_applied,
        "qc_checks": "; ".join(
            f"{c.label}: {c.value_str} ({'PASS' if c.passed else 'FAIL'})" for c in detection.qc_checks
        ),
        "alpha": detection.alpha,
    }
    pd.DataFrame([detection_record]).to_hdf(output_h5_file, key="detection_result", mode="a")

    # Save null reads (synthetic control supporting read counts) as a Series
    pd.Series(detection.null_reads, name="null_reads", dtype=int).to_frame().to_hdf(
        output_h5_file, key="null_reads", mode="a"
    )

    # ── QC report: compute secondary analyses, render via Jinja2 ──
    logger.info(f"Generating MRD QC report (Jinja2). {qc_html_path=}")

    # Secondary analysis 0 (noisy loci filter only): filtered reads without noisy loci filter
    detection_no_noise = None
    df_tf_no_noise = None
    # thresh_noise_lq_reads already normalised (1.0→None) at the top of this function
    if thresh_noise_lq_reads is not None:
        # Reuse df_features (already has n_lq/n_hq columns); apply only read_filter_query
        df_features_filt_no_noise = df_features.query(read_filter_query)
        df_tf_no_noise, _ = mrd.get_tf_from_filtered_data(
            df_features_filt_no_noise,
            df_signatures_filt,
            plot_results=False,
            title="Filtered reads (no noisy loci filter)",
            denom_ratio=denom_ratio,
        )
        # If the multi-read filter is also active, apply it here too so the QC comparison
        # isolates only the noisy-loci filter's impact (both paths have multi-read applied).
        if thresh_multi_read_pvalue is not None:
            df_features_filt_no_noise, _ = mrd.apply_multi_read_locus_filter(
                df_features_filt_no_noise,
                df_tf_no_noise,
                df_signatures_filt,
                thresh_multi_read_pvalue,
            )
            df_tf_no_noise, _ = mrd.get_tf_from_filtered_data(
                df_features_filt_no_noise,
                df_signatures_filt,
                plot_results=False,
                title="Filtered reads (no noisy loci filter, with multi-read filter)",
                denom_ratio=denom_ratio,
            )
        detection_no_noise = run_detection_analysis(
            df_tf=df_tf_no_noise,
            df_signatures_filt=df_signatures_filt,
            alpha=mrd_report_inputs.alpha,
            lod_fpr=mrd_report_inputs.lod_fpr,
            lod_recall=mrd_report_inputs.lod_recall,
        )

    # Secondary analysis 1: filtered reads + unfiltered signatures
    df_tf_unfilt, df_supporting_reads_per_locus_unfilt = mrd.get_tf_from_filtered_data(
        df_features_filt,
        df_signatures,
        plot_results=False,
        title="Filtered reads, unfiltered signatures",
        denom_ratio=denom_ratio,
    )
    detection_unfilt = run_detection_analysis(
        df_tf=df_tf_unfilt,
        df_signatures_filt=df_signatures,
        alpha=mrd_report_inputs.alpha,
        lod_fpr=mrd_report_inputs.lod_fpr,
        lod_recall=mrd_report_inputs.lod_recall,
    )

    # Secondary analysis 2: unfiltered reads + filtered signatures
    df_tf_unfilt2, df_supporting_reads_per_locus_unfilt2 = mrd.get_tf_from_filtered_data(
        df_features,
        df_signatures_filt,
        plot_results=False,
        title="Unfiltered reads, filtered signatures",
        denom_ratio=1,
    )
    detection_unfilt2 = run_detection_analysis(
        df_tf=df_tf_unfilt2,
        df_signatures_filt=df_signatures_filt,
        alpha=mrd_report_inputs.alpha,
        lod_fpr=mrd_report_inputs.lod_fpr,
        lod_recall=mrd_report_inputs.lod_recall,
    )

    # Save HDF5 tables (secondary)
    for key, val in {
        "df_ctdna_vaf_unfilt_signature_filt_featuremap": df_tf_unfilt,
        "df_ctdna_vaf_filt_signature_unfilt_featuremap": df_tf_unfilt2,
        "df_supporting_reads_per_locus_unfilt_signature_filt_featuremap": df_supporting_reads_per_locus_unfilt,
        "df_supporting_reads_per_locus_filt_signature_unfilt_featuremap": df_supporting_reads_per_locus_unfilt2,
    }.items():
        val.to_hdf(output_h5_file, key=key, mode="a")

    if df_tf_no_noise is not None:
        df_tf_no_noise.to_hdf(output_h5_file, key="df_ctdna_vaf_filt_signature_filt_no_noise_filter", mode="a")

    if df_tf_pre_multi_read is not None:
        df_tf_pre_multi_read.to_hdf(
            output_h5_file, key="df_ctdna_vaf_filt_signature_filt_no_multi_read_filter", mode="a"
        )

    # Render QC HTML
    qc_html = render_qc_report(
        detection=detection,
        detection_unfilt=detection_unfilt,
        detection_unfilt2=detection_unfilt2,
        df_tf_filt=df_tf_filt,
        df_tf_unfilt=df_tf_unfilt,
        df_tf_unfilt2=df_tf_unfilt2,
        df_signatures=df_signatures,
        df_signatures_filt=df_signatures_filt,
        df_features=df_features,
        df_features_filt=df_features_filt,
        df_supporting_reads_per_locus_unfilt=df_supporting_reads_per_locus_unfilt,
        df_supporting_reads_per_locus_unfilt2=df_supporting_reads_per_locus_unfilt2,
        basename=mrd_report_inputs.output_basename,
        signature_filter_query=signature_filter_query,
        read_filter_query=read_filter_query,
        denom_ratio=denom_ratio,
        filt_ratio=filt_ratio,
        plot_sbs_fn=plot_sbs_profile,
        plot_af_fn=mrd.plot_signature_allele_fractions,
        df_supporting_reads_per_locus_filt=df_supporting_reads_per_locus_filt,
        applied_filters=applied_filters,
        inputs_info=inputs_info,
        detection_no_noise=detection_no_noise,
        df_tf_no_noise=df_tf_no_noise,
        thresh_noise_lq_reads=thresh_noise_lq_reads,
        detection_pre_multi_read=detection_pre_multi_read,
        df_tf_pre_multi_read=df_tf_pre_multi_read,
        thresh_multi_read_pvalue=thresh_multi_read_pvalue,
    )
    qc_html_path.write_text(qc_html)

    return results_html_path, qc_html_path


def prepare_data_from_mrd_pipeline(mrd_report_inputs: MrdReportInputs, *, return_dataframes=False):
    """
    mrd_report_inputs contains the following fields:
        intersected_featuremaps_parquet: list[str]
            list of featuremaps intesected with various signatures
        matched_signatures_vcf_files: list[str]
            File name or a list of file names, signature vcf files of matched signature/s
        control_signatures_vcf_files: list[str]
            File name or a list of file names, signature vcf files of control signature/s
        db_control_signatures_vcf_files: list[str]
            File name or a list of file names, signature vcf files of db (synthetic) control signature/s
        coverage_csv: str
            Coverage csv file generated with gatk "ExtractCoverageOverVcfFiles", disabled (None) by default
        tumor_sample: str
            sample name in the vcf to take allele fraction (AF) from.
        output_dir: str
            path to which output will be written if not None (default None)
        output_basename: str
            basename of output file (if output_dir is not None must also be not None), default None

    Returns
    -------
    dataframe: pd.DataFrame
        merged data for MRD analysis

    Raises
    -------
    OSError
        in case the file already exists and function executed with no force overwrite
    ValueError
        may be raised
    """
    logger.info(f"Preparing data from MRD pipeline. {mrd_report_inputs=}")
    matched_exists = (
        mrd_report_inputs.matched_signatures_vcf_files is not None
        and len(mrd_report_inputs.matched_signatures_vcf_files) > 0
    )
    control_exists = (
        mrd_report_inputs.control_signatures_vcf_files is not None
        and len(mrd_report_inputs.control_signatures_vcf_files) > 0
    )
    db_control_exists = (
        mrd_report_inputs.db_control_signatures_vcf_files is not None
        and len(mrd_report_inputs.db_control_signatures_vcf_files) > 0
    )

    if mrd_report_inputs.output_dir is not None and mrd_report_inputs.output_basename is None:
        raise ValueError(f"output_dir is not None ({mrd_report_inputs.output_dir}) but output_basename is")
    if mrd_report_inputs.output_dir is not None:
        os.makedirs(mrd_report_inputs.output_dir, exist_ok=True)
    if not matched_exists and not control_exists and not db_control_exists:
        raise ValueError("No signatures files were provided")

    intersection_dataframe_fname = (
        pjoin(
            mrd_report_inputs.output_dir, f"{mrd_report_inputs.output_basename}.features{FileExtension.PARQUET.value}"
        )
        if mrd_report_inputs.output_dir is not None
        else None
    )
    signatures_dataframe_fname = (
        pjoin(
            mrd_report_inputs.output_dir, f"{mrd_report_inputs.output_basename}.signatures{FileExtension.PARQUET.value}"
        )
        if mrd_report_inputs.output_dir is not None
        else None
    )
    # Remove stale parquets from previous runs so read_signature / read_intersection_dataframes
    # don't crash on concat_to_existing_output_parquet=False.
    for _fname in (signatures_dataframe_fname, intersection_dataframe_fname):
        if _fname and os.path.isfile(_fname):
            logger.info("Removing stale output parquet from previous run: %s", _fname)
            os.remove(_fname)

    if matched_exists:
        signature_dataframe = mrd.read_signature(
            mrd_report_inputs.matched_signatures_vcf_files,
            coverage_bed=mrd_report_inputs.coverage_bed,
            output_parquet=signatures_dataframe_fname,
            tumor_sample=mrd_report_inputs.tumor_sample,
            signature_type="matched",
            return_dataframes=return_dataframes,
            concat_to_existing_output_parquet=False,
        )
    if control_exists:
        concat_to_existing_output_parquet = bool(matched_exists)
        signature_dataframe = mrd.read_signature(
            mrd_report_inputs.control_signatures_vcf_files,
            coverage_bed=mrd_report_inputs.coverage_bed,
            output_parquet=signatures_dataframe_fname,
            tumor_sample=mrd_report_inputs.tumor_sample,
            signature_type="control",
            concat_to_existing_output_parquet=concat_to_existing_output_parquet,
        )
    if db_control_exists:
        concat_to_existing_output_parquet = bool(matched_exists or control_exists)
        signature_dataframe = mrd.read_signature(
            mrd_report_inputs.db_control_signatures_vcf_files,
            coverage_bed=mrd_report_inputs.coverage_bed,
            output_parquet=signatures_dataframe_fname,
            tumor_sample=mrd_report_inputs.tumor_sample,
            signature_type="db_control",
            return_dataframes=return_dataframes,
            concat_to_existing_output_parquet=concat_to_existing_output_parquet,
        )

    intersection_dataframe = mrd.read_intersection_dataframes(
        mrd_report_inputs.intersected_featuremaps_parquet,
        output_parquet=intersection_dataframe_fname,
        return_dataframes=return_dataframes,
    )

    if return_dataframes:
        return signature_dataframe, intersection_dataframe
    return signatures_dataframe_fname, intersection_dataframe_fname


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="prepare_data", description="Prepare data for MRD report generation")
    parser.add_argument(
        "-f",
        "--intersected-featuremaps",
        nargs="+",
        type=str,
        required=True,
        help="Input signature and featuemaps vcf files",
    )
    parser.add_argument(
        "--matched-signatures-vcf",
        nargs="+",
        type=str,
        default=None,
        help="Input signature vcf file/s (matched)",
    )
    parser.add_argument(
        "--control-signatures-vcf",
        nargs="+",
        type=str,
        default=None,
        help="Input signature vcf file/s (control)",
    )
    parser.add_argument(
        "--db-control-signatures-vcf",
        nargs="+",
        type=str,
        default=None,
        help="Input signature vcf file/s (db control)",
    )
    parser.add_argument(
        "--coverage-bed",
        type=str,
        default=None,
        required=False,
        help="Coverage bed file generated with mosdepth",
    )
    parser.add_argument(
        "--tumor-sample",
        type=str,
        required=False,
        default=None,
        help=""" sample name in the vcf to take allele fraction (AF) from. Checked with "a in b" so it doesn't have to
    be the full sample name, but does have to return a unique result. Default: None (auto-discovered) """,
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="""Path to which output files will be written.""",
    )
    parser.add_argument(
        "-b",
        "--output-basename",
        type=str,
        default=None,
        help="""Basename of output files that will be created.""",
    )

    parser.add_argument("--signature-filter-query", type=str, default=None, help="Filter query for signatures")
    parser.add_argument("--read-filter-query", type=str, default=None, help="Filter query for reads")
    parser.add_argument("--featuremap-file", type=str, default=None, help="Path to featuremap_df_file")
    parser.add_argument("--srsnv-metadata-json", type=str, default=None, help="Path to srsnv metadata json file")
    parser.add_argument(
        "--thresh-noise-lq-reads",
        type=float,
        nargs="?",
        const=None,
        default=DEFAULT_THRESH_NOISE_LQ_READS,
        help=(
            "Noisy loci filter: remove loci where more than this fraction of reads "
            "fail the read-filter-query. Must be in (0, 1]. "
            f"Default: {DEFAULT_THRESH_NOISE_LQ_READS}. "
            "Pass 1.0 or pass without a value to disable."
        ),
    )
    parser.add_argument(
        "--thresh-multi-read-pvalue",
        type=float,
        nargs="?",
        const=None,
        default=DEFAULT_THRESH_MULTI_READ_PVALUE,
        help=(
            "Multi-read locus filter: remove matched loci whose Bonferroni-corrected "
            "Poisson p-value (P(X >= k | Poisson(TF * mean_coverage)) * signature_size) "
            "falls below this threshold. Targets germline/mosaic variants with unexpectedly "
            f"many supporting reads. Must be ≥ 0. Default: {DEFAULT_THRESH_MULTI_READ_PVALUE}. "
            "Pass 0.0 or omit a value to disable "
            "(e.g. --thresh-multi-read-pvalue 0 or --thresh-multi-read-pvalue with no argument)."
        ),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=("Significance threshold (alpha) for the MRD detection call. " f"Default: {DEFAULT_ALPHA}."),
    )
    parser.add_argument(
        "--lod-fpr",
        type=float,
        default=DEFAULT_LOD_FPR,
        help=("False-positive rate used for personal LOD estimation. " f"Default: {DEFAULT_LOD_FPR}."),
    )
    parser.add_argument(
        "--lod-recall",
        type=float,
        default=DEFAULT_LOD_RECALL,
        help=("Target recall used for personal LOD estimation. " f"Default: {DEFAULT_LOD_RECALL}."),
    )
    args = parser.parse_args(argv[1:])
    if args.thresh_noise_lq_reads is not None and not (0 < args.thresh_noise_lq_reads <= 1):
        parser.error("--thresh-noise-lq-reads must be in the range (0, 1]; use 1.0 or pass without a value to disable")
    # Convert 1.0 → None: threshold of 1.0 can never filter anything; skip the computation
    if args.thresh_noise_lq_reads is not None and args.thresh_noise_lq_reads >= 1.0:
        args.thresh_noise_lq_reads = None
    if args.thresh_multi_read_pvalue is not None and args.thresh_multi_read_pvalue < 0:
        parser.error("--thresh-multi-read-pvalue must be >= 0; use 0 or omit a value to disable")
    # Convert 0.0 → None so the filter is skipped entirely rather than running to produce nothing
    if args.thresh_multi_read_pvalue == 0.0:
        args.thresh_multi_read_pvalue = None
    return args


def main(argv: list[str] | None = None):
    if argv is None:
        argv: list[str] = sys.argv
    args_in = parse_args(argv)

    mrd_report_inputs = MrdReportInputs(
        intersected_featuremaps_parquet=args_in.intersected_featuremaps,
        matched_signatures_vcf_files=args_in.matched_signatures_vcf,
        control_signatures_vcf_files=args_in.control_signatures_vcf,
        db_control_signatures_vcf_files=args_in.db_control_signatures_vcf,
        coverage_bed=args_in.coverage_bed,
        tumor_sample=args_in.tumor_sample,
        output_dir=args_in.output_dir,
        output_basename=args_in.output_basename,
        featuremap_file=args_in.featuremap_file,
        srsnv_metadata_json=args_in.srsnv_metadata_json,
        signature_filter_query=args_in.signature_filter_query,
        read_filter_query=args_in.read_filter_query,
        alpha=args_in.alpha,
        lod_fpr=args_in.lod_fpr,
        lod_recall=args_in.lod_recall,
        thresh_noise_lq_reads=args_in.thresh_noise_lq_reads,
        thresh_multi_read_pvalue=args_in.thresh_multi_read_pvalue,
    )

    results_html, qc_html = generate_mrd_report(mrd_report_inputs)
    logger.info(f"Reports generated: {results_html=}, {qc_html=}")


if __name__ == "__main__":
    main()
