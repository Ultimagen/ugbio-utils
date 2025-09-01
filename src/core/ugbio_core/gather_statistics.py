import argparse
import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from ugbio_core.logger import logger


def get_parser():
    parser = argparse.ArgumentParser(prog="gather_statistics.py", description=run.__doc__)
    parser.add_argument(
        "--stats-txt",
        type=str,
        required=True,
        help="stat input text file",
    )
    parser.add_argument(
        "--barcode",
        type=str,
        required=True,
        help="barcode input file",
    )
    parser.add_argument(
        "--gene-features-stats",
        type=str,
        required=True,
        help="gene features stats input file",
    )
    parser.add_argument(
        "--gene-summary-csv",
        type=str,
        required=True,
        help="gene summary input csv file",
    )
    parser.add_argument(
        "--star-log",
        type=str,
        required=True,
        help="star input log file",
    )
    parser.add_argument(
        "--separate-reads-statistics-json",
        type=str,
        required=False,
        help="separate reads statistics json",
    )
    parser.add_argument("--out-file", default="stat_summary.csv", help="Path to the output csv file.")

    return parser


def run(argv):
    """Intersect BED regions with the option to subtract exclude regions."""

    parser = get_parser()
    args = parser.parse_args(argv[1:])
    stats_to_csv(
        args.stats_txt,
        args.barcode,
        args.gene_features_stats,
        args.gene_summary_csv,
        args.star_log,
        args.out_file,
        args.separate_reads_statistics_json,
    )


def _read_stats_file(path, columns=("Measure", "Value"), sep=r"\s+", header=None, stat_type=None, skiprows=None):
    """Read a generic stats file into a standardized DataFrame."""
    stat_df = pd.read_csv(path, sep=sep, header=header, skiprows=skiprows)
    stat_df.columns = columns
    if stat_type:
        stat_df["Stat_Type"] = stat_type
    return stat_df


def _add_percentage_column(stat_df, condition_func, value_col="Value"):
    """Add a Percentage column if condition_func holds for a row."""
    stat_df["Percentage"] = np.nan
    mask = condition_func(stat_df)
    stat_df.loc[mask, "Percentage"] = (stat_df.loc[mask, value_col].astype(float) * 100).astype(str) + "%"
    stat_df.loc[mask, value_col] = np.nan
    return stat_df


def _process_summary_csv(path):
    stat_df = _read_stats_file(path, sep=",", stat_type="Summary")
    is_fraction = stat_df["Value"].astype(float) <= 1
    return _add_percentage_column(stat_df, lambda d: is_fraction)


def _process_star_log(path):
    stat_df = _read_stats_file(path, sep="\t")
    stat_df.columns = ["Measure", "Value"]
    stat_df = stat_df[stat_df["Measure"].str.endswith("|")]
    stat_df["Measure"] = stat_df["Measure"].str.replace(r" \|\s*", "", regex=True)
    stat_df["Percentage"] = np.nan
    mask = stat_df["Value"].astype(str).str.endswith("%")
    stat_df.loc[mask, "Percentage"] = stat_df.loc[mask, "Value"]
    stat_df.loc[mask, "Value"] = np.nan
    stat_df["Stat_Type"] = "Alignment"
    return stat_df


def _process_adapter_json(path):
    with open(path) as f:
        data = json.load(f)

    metrics = defaultdict(int)
    metrics.update(
        {
            "Total read pairs processed": data["readsIn"],
            "Reads written (passing filters)": int(data["readsOut"].split()[0]),
            "Reads filtered": int(data["filtered"].split()[0]),
            "Reads with 5p adapter": int(data["adapter5p"].split()[0]),
            "Reads with 3p adapter": int(data["adapter3p"].split()[0]),
            "Reads with middle sequence": int(data["adapterMiddle"].split()[0]),
            "Total base pairs processed": data["bpIn"],
            "Total base pairs output": int(data["bpOut"].split()[0]),
            "Reads too short after quality trimming": int(data["trimmedTooShort"].split()[0]),
            "Discarded base pairs, quality trimming": int(data["bpCutoff"].split()[0]),
            "Reads with low umi quality": int(data["umiQualityDropped"].split()[0]),
            "Read 2 too short": int(data["read2TooShortDropped"].split()[0]),
            "Read 1 too short": int(data["read1TooShortDropped"].split()[0]),
        }
    )

    stat_df = pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"]).reset_index()
    stat_df = stat_df.rename(columns={"index": "Measure"})
    stat_df["Stat_Type"] = "Adapter removal and prefiltering"
    stat_df["Percentage"] = np.nan

    # Percentages relative to totals
    totals = metrics["Total read pairs processed"]
    filtered = metrics["Reads filtered"]

    stat_df.loc[
        stat_df["Measure"].isin(
            [
                "Reads written (passing filters)",
                "Reads too short after quality trimming",
                "Reads with low umi quality",
                "Read 1 too short",
                "Read 2 too short",
            ]
        ),
        "Percentage",
    ] = 100 * stat_df["Value"] / totals

    stat_df.loc[
        stat_df["Measure"].isin(["Reads with 5p adapter", "Reads with middle sequence", "Reads with 3p adapter"]),
        "Percentage",
    ] = 100 * stat_df["Value"] / filtered

    stat_df.loc[stat_df["Measure"] == "Total base pairs output", "Percentage"] = (
        100 * stat_df["Value"] / metrics["Total base pairs processed"]
    )

    return stat_df


def stats_to_csv(
    stats_txt: str,
    barcode: str,
    gene_features_stats: str,
    gene_summary_csv: str,
    star_log: str,
    out_file: str,
    separate_reads_statistics_json: str = None,
):
    """Combine various QC stats files into one normalized CSV."""

    logger.info("Loading stats files...")

    param_df = _read_stats_file(stats_txt, stat_type="Params", skiprows=1)
    param_df["Measure"] = param_df["Measure"].str.replace(":$", "", regex=True)

    barcode_df = _read_stats_file(barcode, stat_type="Barcode")
    feature_df = _read_stats_file(gene_features_stats, stat_type="Feature")
    summary_df = _process_summary_csv(gene_summary_csv)
    alignment_df = _process_star_log(star_log)

    adapter_df = (
        _process_adapter_json(separate_reads_statistics_json) if separate_reads_statistics_json else pd.DataFrame()
    )

    # TODO: add CBCUMI adapter stats if needed
    adapter_cbcumi_df = pd.DataFrame()

    logger.info("Combining dataframes...")

    combined_df = pd.concat(
        [param_df, adapter_df, adapter_cbcumi_df, alignment_df, feature_df, barcode_df, summary_df],
        ignore_index=True,
    )

    # Post-processing
    combined_df["Measure"] = combined_df["Measure"].str.strip()
    combined_df["index_orig"] = combined_df.index

    combined_df = (
        combined_df[["index_orig", "Stat_Type", "Measure", "Value", "Percentage"]]
        .melt(
            id_vars=["index_orig", "Stat_Type", "Measure"],
            var_name="Value_Type",
            value_name="Measure_Value",
        )
        .dropna()
    )

    # Fix date-like values
    combined_df.loc[combined_df["Measure"].str.endswith(" on"), "Value_Type"] = "Date"

    # Convert percentages to numeric
    mask = combined_df["Measure_Value"].astype(str).str.endswith("%")
    combined_df.loc[mask, "Measure_Value"] = combined_df.loc[mask, "Measure_Value"].str[:-1].astype(float)

    logger.info("Writing combined CSV...")
    combined_df.to_csv(out_file, index=False)


def main():
    run(sys.argv)


if __name__ == "__main__":
    main()
