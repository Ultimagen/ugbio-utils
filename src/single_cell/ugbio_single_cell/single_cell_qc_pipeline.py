from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from ugbio_core.reports.report_utils import generate_report
from ugbio_single_cell.collect_statistics import (
    collect_statistics,
    extract_statistics_table,
)
from ugbio_single_cell.create_plots import (
    cbc_umi_plot,
    plot_insert_length_histogram,
    plot_mean_insert_quality_histogram,
    plot_quality_per_position,
)
from ugbio_single_cell.sc_qc_dataclasses import (
    TEMPLATE_NOTEBOOK,
    H5Keys,
    Inputs,
    OutputFiles,
    Thresholds,
)


def single_cell_qc(
    input_files: Inputs,
    output_path: str,
    thresholds: Thresholds,
    sample_name: str,
    star_db: str = "STAR_hg38_3_2.7.10a",
    *,
    plot_barcode_rank: bool = False,
):
    """
    Run single cell qc pipeline that collects statistics, prepares parameters for report and generates report

    Parameters
    ----------
    input_files : Inputs
        Inputs object with paths to input files
    output_path : str
        Path to output directory
    thresholds : Thresholds
        Thresholds object with thresholds for qc
    sample_name : str
        Sample name to be included as a prefix in the output files
    star_db : str
        DB name used when running STAR
    plot_barcode_rank : bool
        Plot barcode rank plot
    """
    if not sample_name.endswith("."):
        sample_name += "."

    h5_file = collect_statistics(
        input_files, output_path, sample_name, star_db, save_trimmer_histogram=plot_barcode_rank
    )
    extract_statistics_table(h5_file)

    params, tmp_files = prepare_parameters_for_report(
        h5_file, thresholds, output_path, plot_barcode_rank=plot_barcode_rank
    )
    generate_single_cell_report(params, output_path, tmp_files, sample_name)

    # keep only STAR and short table data in h5 file
    with pd.HDFStore(h5_file, "a") as store:
        keys_to_keep = [
            H5Keys.STATISTICS_SHORTLIST.value,
            H5Keys.STAR_STATS.value,
        ]
        for key in store.keys():
            if key.strip("/") not in keys_to_keep:
                store.remove(key)

    # keys to convert to json to disply in pyprus
    keys_to_convert_to_json = pd.Series([H5Keys.STATISTICS_SHORTLIST.value, H5Keys.STAR_STATS.value])
    keys_to_convert_to_json.to_hdf(h5_file, key="keys_to_convert")


def prepare_parameters_for_report(
    h5_file: Path, thresholds: Thresholds, output_path: str, *, plot_barcode_rank: bool = False
) -> tuple[dict, list[Path]]:
    """
    Prepare parameters for report generation (h5 file, thresholds, plots)

    Parameters
    ----------
    h5_file : Path
        Path to h5 file with statistics
    thresholds : Thresholds
        Thresholds object with thresholds for qc
    output_path : str
        Path to output directory
    plot_barcode_rank : bool
        Plot barcode rank plot

    Returns
    -------
    tuple[dict, list[Path]]
        Parameters for report, list of temporary files to be removed after report generation
    """
    # list of files to be removed after report generation
    tmp_files = []

    # prepare parameters for report: add statistics
    parameters = {"statistics_h5": h5_file}

    # add thresholds to parameters
    for threshold_name, threshold_value in vars(thresholds).items():
        parameters[threshold_name + "_threshold"] = threshold_value

    # add plots to parameters
    if plot_barcode_rank:
        try:
            cbc_umi_png = cbc_umi_plot(h5_file, output_path)
            parameters["cbc_umi_png"] = cbc_umi_png
            tmp_files.append(cbc_umi_png)
        except Exception as e:
            print(f"Failed to plot barcode rank plot: {e}")

    insert_length_png = plot_insert_length_histogram(h5_file, output_path)
    parameters["insert_length_png"] = insert_length_png
    tmp_files.append(insert_length_png)

    mean_insert_quality_histogram_png = plot_mean_insert_quality_histogram(h5_file, output_path)
    parameters["mean_insert_quality_histogram_png"] = mean_insert_quality_histogram_png
    tmp_files.append(mean_insert_quality_histogram_png)

    quality_per_position_png = plot_quality_per_position(h5_file, output_path)
    parameters["quality_per_position_png"] = quality_per_position_png
    tmp_files.append(quality_per_position_png)

    return parameters, tmp_files


def generate_single_cell_report(parameters, output_path, tmp_files: list[Path], sample_name: str) -> Path:
    """
    Generate report based on jupyter notebook template.

    Parameters
    ----------
    parameters : dict
        Parameters for report
    output_path : str
        Path to output directory
    tmp_files : list[Path]
        List of temporary files to be removed after report generation
    sample_name : str
        Sample name to be included as a prefix in the output file

    Returns
    -------
    Path
        Path to generated report
    """
    output_report_html = Path(output_path) / (sample_name + OutputFiles.HTML_REPORT.value)

    generate_report(
        template_notebook_path=TEMPLATE_NOTEBOOK,
        parameters=parameters,
        output_report_html_path=output_report_html,
        tmp_files=tmp_files,
    )

    return output_report_html


def main():
    # parse args from command line
    parser = ArgumentParser()
    parser.add_argument(
        "--sample-name",
        type=str,
        required=True,
        help="Sample name to be included in the output files",
    )
    parser.add_argument(
        "--trimmer-stats",
        type=str,
        required=True,
        help="Path to Trimmer stats csv file",
    )
    parser.add_argument(
        "--trimmer-histogram",
        type=str,
        required=False,
        nargs="+",
        help="Path to Trimmer histogram csv files. Multiple files are supported, pass them with space separated.",
    )
    parser.add_argument(
        "--trimmer-failure-codes",
        type=str,
        required=True,
        help="Path to Trimmer failure codes csv file",
    )
    parser.add_argument(
        "--sorter-stats",
        type=str,
        required=True,
        help="Path to Sorter stats csv file",
    )
    parser.add_argument(
        "--sorter-stats-json",
        type=str,
        required=False,
        help="Path to Sorter stats json file",
    )
    parser.add_argument("--star-stats", type=str, required=True, help="Path to STAR stats file")
    parser.add_argument(
        "--insert",
        type=str,
        required=True,
        help="Path to insert .fastq.gz file",
    )
    parser.add_argument(
        "--star-reads-per-gene",
        type=str,
        required=True,
        help="Path to STAR ReadsPerGene.out.tab file",
    )
    parser.add_argument("--output-path", type=str, required=True, help="Path to output directory")
    parser.add_argument("--pass-trim-rate", type=float, required=True, help="Minimal %trimmed")
    parser.add_argument("--read-length", type=int, required=True, help="Expected read length")
    parser.add_argument(
        "--fraction-below-read-length",
        type=float,
        required=True,
        help="Fraction of reads below read length threshold",
    )
    parser.add_argument(
        "--percent-aligned",
        type=float,
        required=True,
        help="Minimal % of reads aligned",
    )

    parser.add_argument("--plot-bracode-rank", type=bool, default=False, help="Plot barcode rank plot")
    parser.add_argument("--star-db", type=str, default="STAR_hg38_3_2.7.10a", help="DB name used when running STAR")

    args = parser.parse_args()

    # create Inputs and Thresholds objects
    inputs = Inputs(
        trimmer_stats_csv=args.trimmer_stats,
        trimmer_histogram_csv=args.trimmer_histogram,
        trimmer_failure_codes_csv=args.trimmer_failure_codes,
        sorter_stats_csv=args.sorter_stats,
        sorter_stats_json=args.sorter_stats_json,
        star_stats=args.star_stats,
        star_reads_per_gene=args.star_reads_per_gene,
        insert=args.insert,
    )
    thresholds = Thresholds(
        args.pass_trim_rate,
        args.read_length,
        args.fraction_below_read_length,
        args.percent_aligned,
    )
    # run single_cell_qc
    single_cell_qc(
        input_files=inputs,
        output_path=args.output_path,
        thresholds=thresholds,
        sample_name=args.sample_name,
        star_db=args.star_db,
        plot_barcode_rank=args.plot_bracode_rank,
    )


if __name__ == "__main__":
    main()
