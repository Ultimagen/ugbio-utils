import os
from os.path import basename
from os.path import join as pjoin

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from ugbio_core.logger import logger


def generate_stats_from_histogram(
    val_count,
    quantiles=np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.95]),
    out_path=None,
    *,
    verbose=True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(val_count, str) and os.path.isfile(val_count):
        val_count = pd.read_hdf(val_count, key="histogram")
    if val_count.shape[0] == 0:  # empty input
        raise ValueError("Empty dataframe - most likely bed files were not created, bam/cram index possibly missing")
    df_percentiles = pd.concat(
        (
            val_count.apply(lambda x: interp1d(np.cumsum(x), val_count.index, bounds_error=False)(quantiles)),
            val_count.apply(
                lambda x: np.sum(val_count.index.to_numpy() * x.to_numpy())
                / np.sum(x.to_numpy())  # np.average gave strange bug
                if not x.isna().all() and x.sum() > 0
                else np.nan
            )
            .to_frame()
            .T,
        ),
        sort=False,
    ).fillna(0)  # extrapolation below 0 yields NaN
    df_percentiles.index = pd.Index(data=[f"Q{int(qq * 100)}" for qq in quantiles] + ["mean"], name="statistic")
    genome_median = df_percentiles.loc["Q50"].filter(regex="Genome").to_numpy()[0]
    selected_percentiles = (
        df_percentiles.loc[[f"Q{q}" for q in (5, 10, 50)]]
        .rename(index={"Q50": "median_coverage"})
        .rename(index={f"Q{q}": f"percentile_{q}" for q in (5, 10, 50)})
    )
    selected_percentiles.loc["median_coverage_normalized"] = selected_percentiles.loc["median_coverage"] / genome_median
    df_stats = pd.concat(
        (
            selected_percentiles,
            pd.concat(
                (
                    (val_count[val_count.index >= (genome_median * 0.5)] * 100)
                    .sum()
                    .rename("percent_larger_than_05_of_genome_median")
                    .to_frame()
                    .T,
                    (val_count[val_count.index >= (genome_median * 0.25)] * 100)
                    .sum()
                    .rename("percent_larger_than_025_of_genome_median")
                    .to_frame()
                    .T,
                    (val_count[val_count.index >= 10] * 100).sum().rename("percent_over_or_equal_to_10x").to_frame().T,  # noqa PLR2004
                    (val_count[val_count.index >= 20] * 100).sum().rename("percent_over_or_equal_to_20x").to_frame().T,  # noqa PLR2004
                )
            ),
        )
    )
    if verbose:
        logger.debug("Generated stats:\n%s", df_stats.iloc[:, :10].to_string())

    if out_path is not None:
        os.makedirs(out_path, exist_ok=True)
        logger.debug("Saving data")
        if "." in basename(out_path):
            coverage_stats_dataframes = out_path
        else:
            coverage_stats_dataframes = pjoin(out_path, "coverage_stats.h5")
        if verbose:
            logger.debug("Saving dataframes to %s", coverage_stats_dataframes)
        df_stats.to_hdf(coverage_stats_dataframes, key="stats", mode="a")
        df_percentiles.to_hdf(coverage_stats_dataframes, key="percentiles", mode="a")
        return coverage_stats_dataframes

    return df_percentiles, df_stats
