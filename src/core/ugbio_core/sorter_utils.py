import json

import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
from ugbio_core.plotting_utils import set_pyplot_defaults


def read_sorter_statistics_csv(sorter_stats_csv: str, *, edit_metric_names: bool = True) -> pd.Series:
    """
    Collect sorter statistics from csv

    Parameters
    ----------
    sorter_stats_csv : str
        path to a Sorter stats file
    edit_metric_names: bool
        if True, edit the metric names to be consistent in the naming of percentages

    Returns
    -------
    pd.Series
        Series with sorter statistics
    """

    # read Sorter stats
    df_sorter_stats = pd.read_csv(sorter_stats_csv, header=None, names=["metric", "value"]).set_index("metric")
    # replace '(' and ')' in values (legacy format for F95)
    df_sorter_stats = df_sorter_stats.assign(
        value=df_sorter_stats["value"]
        .astype(str)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .astype(float)
        .values
    )
    # add convenient metric
    if "Failed_QC_reads" in df_sorter_stats.index and "PF_Barcode_reads" in df_sorter_stats.index:
        df_sorter_stats.loc["PCT_Failed_QC_reads"] = (
            100
            * df_sorter_stats.loc["Failed_QC_reads"]
            / (df_sorter_stats.loc["Failed_QC_reads"] + df_sorter_stats.loc["PF_Barcode_reads"])
        )

    if edit_metric_names:
        # rename metrics to uniform convention
        df_sorter_stats = df_sorter_stats.rename({c: c.replace("% ", "PCT_") for c in df_sorter_stats.index})
    return df_sorter_stats["value"]


def read_and_parse_sorter_statistics_csv(sorter_stats_csv: str, metrics_shortlist: list = None) -> pd.Series:
    """
    Read and parse a sorter statistics csv file

    Parameters
    ----------
    sorter_stats_csv : str
        path to a Sorter stats file

    Returns
    -------
    pd.Series
        Series with sorter statistics
    """
    df_sorter_stats = read_sorter_statistics_csv(sorter_stats_csv)

    # create statistics shortlist
    if metrics_shortlist is None:
        metrics_shortlist = [
            "Mean_cvg",
            "Indel_Rate",
            "Mean_Read_Length",
            "PCT_PF_Reads_aligned",
            "PCT_Chimeras",
            "PCT_duplicates",
            "PF_Barcode_reads",
        ]
    df_stats_shortlist = df_sorter_stats.reindex(metrics_shortlist)

    return df_stats_shortlist


def read_effective_coverage_from_sorter_json(
    sorter_stats_json, min_coverage_for_fp=20, max_coverage_percentile=0.95, min_mapq=60
):
    """
    Read effective coverage metrics from sorter JSON file - mean coverage, ratio of reads over MAPQ, ratio of bases in
    the given coverage range, min coverage for FP calculation, coverage of max percentile.

    Parameters
    ----------
    sorter_stats_json : str
        Path to Sorter statistics JSON file.
    min_coverage_for_fp : int
        Minimum coverage to consider for FP calculation.
    max_coverage_percentile : float
        Maximum coverage percentile to consider for FP calculation.
    min_mapq : int
        Minimum MAPQ for reads to be included

    Returns
    -------
    tuple
        (mean_coverage, ratio_of_reads_over_mapq, ratio_of_bases_in_coverage_range,
         min_coverage_for_fp, coverage_of_max_percentile)

    """
    with open(sorter_stats_json, encoding="utf-8") as fh:
        sorter_stats = json.load(fh)

    # Calculate ratio_of_bases_in_coverage_range
    cvg = pd.Series(sorter_stats["base_coverage"].get("Genome", sorter_stats["cvg"]))
    cvg_cdf = cvg.cumsum() / cvg.sum()
    ratio_below_min_coverage = cvg_cdf.loc[min_coverage_for_fp]

    if ratio_below_min_coverage > 0.5:  # noqa: PLR2004
        min_coverage_for_fp = (cvg_cdf >= 0.5).argmax()  # noqa: PLR2004
    coverage_of_max_percentile = (cvg_cdf >= max_coverage_percentile).argmax()

    ratio_of_bases_in_coverage_range = cvg_cdf[coverage_of_max_percentile] - cvg_cdf[min_coverage_for_fp]

    # Calculate ratio_of_reads_over_mapq
    reads_by_mapq = pd.Series(sorter_stats["mapq"])
    ratio_of_reads_over_mapq = reads_by_mapq[reads_by_mapq.index >= min_mapq].sum() / reads_by_mapq.sum()

    # Calculate mean coverage
    mean_coverage = (cvg.index.to_numpy() * cvg.to_numpy()).sum() / cvg.sum()

    return (
        mean_coverage,
        ratio_of_reads_over_mapq,
        ratio_of_bases_in_coverage_range,
        min_coverage_for_fp,
        coverage_of_max_percentile,
    )


def plot_read_length_histogram(
    sorter_stats_json: str,
    plot_range_percentiles: tuple = (0.001, 0.999),
    output_filename: str = None,
    title: str = None,
):
    """
    Plot read length histogram from sorter JSON file.

    Parameters
    ----------
    sorter_stats_json : str
        Path to Sorter statistics JSON file.
    plot_range_percentiles : tuple
        Percentiles to show in plot range. Default: (0.01, 0.99)
    output_filename : str
        Output file name. If None (default), don't save.
    title : str
        Plot title. If None (default), don't show title.
    """
    set_pyplot_defaults()

    # read histograms
    read_length = get_histogram_from_sorter(sorter_stats_json, "read_length")
    aligned_read_length = get_histogram_from_sorter(sorter_stats_json, "aligned_read_length")
    # calculate pdf, cdf and limits
    pdf_aligned = aligned_read_length / read_length.sum()
    cdf_aligned = pdf_aligned.cumsum()
    pdf_aligned_to_all_reads = aligned_read_length / read_length.sum()
    pdf = read_length / read_length.sum()
    cdf = pdf.cumsum()
    f_interp = interp1d(cdf.values, cdf.index, kind="linear", fill_value=(0, 1))
    f_interp_aligned = interp1d(cdf_aligned.values, cdf_aligned.index, kind="linear", fill_value=(0, 1))
    xlim = f_interp(plot_range_percentiles)
    # plot
    plt.figure(figsize=(10, 5))
    pdf.plot(
        linewidth=3,
    )
    pdf_aligned_to_all_reads.plot(linewidth=2, linestyle="--")
    legend_handle = plt.legend(
        [
            f"All reads, median={f_interp(0.5):.0f}",
            f"Aligned reads, median={f_interp_aligned(0.5):.0f}",
        ]
    )
    plt.xlim(xlim)
    plt.xlabel("Read length")
    bbox_extra_artists = [legend_handle]
    if title:
        title_handle = plt.title(title)
        bbox_extra_artists.append(title_handle)
    # save
    if output_filename:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=bbox_extra_artists,
        )


def get_histogram_from_sorter(sorter_stats_json: str, histogram_key: str) -> pd.DataFrame:
    """
    Read histogram from sorter JSON file.

    Parameters
    ----------
    sorter_stats_json : str
        Path to Sorter statistics JSON file.
    histogram_key : str
        Histogram key to read from sorter JSON file.
        Allowed values: 'bqual', 'pf_bqual', 'read_length', 'aligned_read_length', 'mapq', 'cvg', 'bqx'

    Returns
    -------
    pd.DataFrame
        Histogram data.

    Raises
    ------
    ValueError
        If histogram_key is not one of the allowed values.

    """
    allowed_values = (
        "bqual",
        "pf_bqual",
        "read_length",
        "aligned_read_length",
        "mapq",
        "cvg",
        "bqx",
    )  # known histogram keys in sorter JSON
    if histogram_key not in allowed_values:
        raise ValueError(f"histogram_key must be one of {allowed_values}, got {histogram_key}")
    with open(sorter_stats_json, encoding="utf-8") as fh:
        sorter_stats = json.load(fh)
    histogram = pd.Series(sorter_stats[histogram_key], name="count")
    histogram.index.name = histogram_key

    return histogram
