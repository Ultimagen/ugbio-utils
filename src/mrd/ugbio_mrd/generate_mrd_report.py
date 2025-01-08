import argparse
import os
import sys
from dataclasses import dataclass
from os.path import join as pjoin
from pathlib import Path

from ugbio_core.consts import FileExtension
from ugbio_core.logger import logger
from ugbio_core.reports.report_utils import generate_report

from ugbio_mrd.mrd_utils import read_intersection_dataframes, read_signature

HTML_REPORT = ".mrd_data_analysis.html"
BASE_PATH = Path(__file__).parent  # should be: src/mrd/ugbio_mrd
TEMPLATE_NOTEBOOK = BASE_PATH / "reports" / "mrd_automatic_data_analysis.ipynb"


@dataclass
class MrdReportInputs:
    """Inputs to generate a MRD report"""

    intersected_featuremaps_parquet: list[str]
    matched_signatures_vcf_files: list[str]
    control_signatures_vcf_files: list[str]
    coverage_csv: str
    output_dir: str
    output_basename: str
    featuremap_file: str
    db_control_signatures_vcf_files: list[str] = None
    tumor_sample: str = None
    signature_filter_query: str = None
    read_filter_query: str = None


def generate_mrd_report(mrd_report_inputs: MrdReportInputs):
    signatures_path, intersection_path = prepare_data_from_mrd_pipeline(mrd_report_inputs)

    parameters = {
        "features_file_parquet": intersection_path,
        "signatures_file_parquet": signatures_path,
        "featuremap_df_file": mrd_report_inputs.featuremap_file,
        "signature_filter_query": mrd_report_inputs.signature_filter_query,
        "read_filter_query": mrd_report_inputs.read_filter_query,
        "output_dir": mrd_report_inputs.output_dir,
        "basename": mrd_report_inputs.output_basename,
    }
    output_report_html_path = Path(mrd_report_inputs.output_dir) / Path(mrd_report_inputs.output_basename + HTML_REPORT)

    logger.info(f"Generating MRD report. {parameters=}, {output_report_html_path=}")

    generate_report(
        template_notebook_path=TEMPLATE_NOTEBOOK, parameters=parameters, output_report_html_path=output_report_html_path
    )
    return output_report_html_path


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
            coverage_csv=mrd_report_inputs.coverage_csv,
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
            coverage_csv=mrd_report_inputs.coverage_csv,
            output_parquet=signatures_dataframe_fname,
            tumor_sample=mrd_report_inputs.tumor_sample,
            signature_type="control",
            concat_to_existing_output_parquet=concat_to_existing_output_parquet,
        )
    if db_control_exists:
        concat_to_existing_output_parquet = bool(matched_exists or control_exists)
        signature_dataframe = read_signature(
            mrd_report_inputs.db_control_signatures_vcf_files,
            coverage_csv=mrd_report_inputs.coverage_csv,
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
        "--coverage-csv",
        type=str,
        default=None,
        required=False,
        help="Coverage csv file generated with gatk ExtractCoverageOverVcfFiles",
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
    parser.add_argument("--featuremap-file", type=str, default=None, help="Path to Featuremap file")
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
        coverage_csv=args_in.coverage_csv,
        tumor_sample=args_in.tumor_sample,
        output_dir=args_in.output_dir,
        output_basename=args_in.output_basename,
        featuremap_file=args_in.featuremap_file,
        signature_filter_query=args_in.signature_filter_query,
        read_filter_query=args_in.read_filter_query,
    )

    generate_mrd_report(mrd_report_inputs)


if __name__ == "__main__":
    main()
