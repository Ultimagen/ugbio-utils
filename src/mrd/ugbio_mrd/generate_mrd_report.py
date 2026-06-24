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
from ugbio_mrd.mrd_detection import run_detection_analysis
from ugbio_mrd.mrd_report_renderer import render_analysis_report, render_qc_report
from ugbio_mrd.mrd_utils import read_intersection_dataframes, read_signature

RESULTS_HTML_REPORT = ".mrd_analysis_report.html"
QC_HTML_REPORT = ".mrd_qc_report.html"
BASE_PATH = Path(__file__).parent  # should be: src/mrd/ugbio_mrd


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


def generate_mrd_report(mrd_report_inputs: MrdReportInputs) -> tuple[Path, Path]:  # noqa: PLR0915
    """
    Generate both the MRD analysis report (Jinja2 HTML) and the MRD QC report (notebook).

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
    read_filter_query = mrd_report_inputs.read_filter_query or "filt>0 and snvq>60 and mapq>=60"

    # 1. Load data
    df_features, df_features_filt, filtering_ratio = mrd.read_and_filter_features_parquet(
        intersection_path, read_filter_query
    )
    df_signatures, df_signatures_filt = mrd.read_and_filter_signatures_parquet(
        signatures_path, signature_filter_query, filtering_ratio
    )
    denom_ratio, filt_ratio, _ = mrd.calc_tumor_fraction_denominator_ratio(
        mrd_report_inputs.featuremap_file, mrd_report_inputs.srsnv_metadata_json, read_filter_query
    )
    df_tf_filt, df_supporting_reads_per_locus_filt = mrd.get_tf_from_filtered_data(
        df_features_filt,
        df_signatures_filt,
        plot_results=False,
        title="Filtered reads and signatures",
        denom_ratio=denom_ratio,
    )

    # 2. Run detection
    detection = run_detection_analysis(
        df_tf=df_tf_filt,
        df_signatures_filt=df_signatures_filt,
        denom_ratio=denom_ratio,
        df_supporting_reads_per_locus=df_supporting_reads_per_locus_filt,
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

    # 4. SBS plot helper
    _sbs_types = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]  # noqa: F841
    _sbs_colors = ["#1EBFF0", "#050708", "#E62725", "#CBCACB", "#A1C935", "#ECC6C5"]

    def plot_sbs_profile(df_sig, title="", ax=None, query=None):
        df = df_sig if query is None else df_sig.query(query)
        _all_muts = ["C->A", "C->G", "C->T", "T->A", "T->C", "T->G"]
        counts = df["mutation_type"].value_counts(normalize=True).reindex(_all_muts, fill_value=0)
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
            0.99, 0.02, f"n = {len(df):,}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#7f8c8d"
        )
        ax.spines[["top", "right"]].set_visible(False)
        return ax

    # 5. Render HTML
    def _fmt_files(files):
        if not files:
            return "—"
        return ", ".join(Path(f).name for f in files)

    inputs_info = {
        "Sample": mrd_report_inputs.output_basename or "N/A",
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
        "detection_threshold": detection.detection_threshold,
        "personal_lod": detection.personal_lod,
        "signature_size": detection.signature_size,
        "mean_coverage": detection.mean_coverage,
        "corrected_coverage": detection.corrected_coverage,
        "qc_checks": [
            {"label": c.label, "value": c.value_str, "threshold": c.threshold_str, "passed": c.passed}
            for c in detection.qc_checks
        ],
        "alpha": 0.01,
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
        "detection_threshold": detection.detection_threshold,
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
        "alpha": 0.01,
    }
    pd.DataFrame([detection_record]).to_hdf(output_h5_file, key="detection_result", mode="a")

    # Save null reads (synthetic control supporting read counts) as a Series
    pd.Series(detection.null_reads, name="null_reads", dtype=int).to_frame().to_hdf(
        output_h5_file, key="null_reads", mode="a"
    )

    # ── QC report: compute secondary analyses, render via Jinja2 ──
    logger.info(f"Generating MRD QC report (Jinja2). {qc_html_path=}")

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
        denom_ratio=denom_ratio,
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
        denom_ratio=1,
    )

    # Save HDF5 tables (secondary)
    for key, val in {
        "df_ctdna_vaf_unfilt_signature_filt_featuremap": df_tf_unfilt,
        "df_ctdna_vaf_filt_signature_unfilt_featuremap": df_tf_unfilt2,
        "df_supporting_reads_per_locus_unfilt_signature_filt_featuremap": df_supporting_reads_per_locus_unfilt,
        "df_supporting_reads_per_locus_filt_signature_unfilt_featuremap": df_supporting_reads_per_locus_unfilt2,
    }.items():
        val.to_hdf(output_h5_file, key=key, mode="a")

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

    intersection_dataframe = read_intersection_dataframes(
        mrd_report_inputs.intersected_featuremaps_parquet,
        output_parquet=intersection_dataframe_fname,
        return_dataframes=True,
    )
    if matched_exists:
        signature_dataframe = read_signature(
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
        signature_dataframe = read_signature(
            mrd_report_inputs.control_signatures_vcf_files,
            coverage_bed=mrd_report_inputs.coverage_bed,
            output_parquet=signatures_dataframe_fname,
            tumor_sample=mrd_report_inputs.tumor_sample,
            signature_type="control",
            concat_to_existing_output_parquet=concat_to_existing_output_parquet,
        )
    if db_control_exists:
        concat_to_existing_output_parquet = bool(matched_exists or control_exists)
        signature_dataframe = read_signature(
            mrd_report_inputs.db_control_signatures_vcf_files,
            coverage_bed=mrd_report_inputs.coverage_bed,
            output_parquet=signatures_dataframe_fname,
            tumor_sample=mrd_report_inputs.tumor_sample,
            signature_type="db_control",
            return_dataframes=return_dataframes,
            concat_to_existing_output_parquet=concat_to_existing_output_parquet,
        )

    intersection_dataframe = read_intersection_dataframes(
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
    return parser.parse_args(argv[1:])


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
    )

    results_html, qc_html = generate_mrd_report(mrd_report_inputs)
    logger.info(f"Reports generated: {results_html=}, {qc_html=}")


if __name__ == "__main__":
    main()
