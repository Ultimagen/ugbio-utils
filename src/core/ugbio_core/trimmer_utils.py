import os

import pandas as pd


def merge_trimmer_histograms(trimmer_histograms: list[str], output_path: str):
    """
    Merge multiple Trimmer histograms into a single histogram.

    Parameters
    ----------
    trimmer_histograms : list[str]
        List of paths to Trimmer histogram files, or a single path to a Trimmer histogram file.
    output_path : str
        Path to output file, or a path to which the output file will be written to with the basename of the first
        value in trimmer_histograms.

    Returns
    -------
    str
        Path to output file. If the list is only 1 file, returns the path to that file without doing anything.

    Raises
    ------
    ValueError
        If trimmer_histograms is empty.
    """
    if len(trimmer_histograms) == 0:
        raise ValueError("trimmer_histograms must not be empty")
    if isinstance(trimmer_histograms, str):
        trimmer_histograms = [trimmer_histograms]
    if len(trimmer_histograms) == 1:
        return trimmer_histograms[0]

    # read and merge histograms
    df_concat = pd.concat(pd.read_csv(x) for x in trimmer_histograms)
    if df_concat.columns[-1] != "count":
        raise ValueError(f"Unexpected columns in histogram files: {df_concat.columns}")
    df_merged = df_concat.groupby(df_concat.columns[:-1].tolist(), dropna=False).sum().reset_index()
    # write to file
    output_filename = (
        os.path.join(output_path, os.path.basename(trimmer_histograms[0]))
        if os.path.isdir(output_path)
        else output_path
    )
    df_merged.to_csv(output_filename, index=False)
    return output_filename


def read_trimmer_failure_codes(
    trimmer_failure_codes_csv: str, *, add_total: bool = False, include_failed_rsq=False, include_pretrim_filters=True
) -> pd.DataFrame:
    """
    Read a trimmer failure codes csv file

    Parameters
    ----------
    trimmer_failure_codes_csv : str
        path to a Trimmer failure codes file
    add_total : bool
        if True, add a row with total failed reads to the dataframe
    include_failed_rsq : bool
        if True, include failed RSQ reads in the dataframe and in the percentage calculation (default: False)
    include_pretrim_filters: bool
        if True, include pre-trimming failure reasons encoded in start segment (default: True)

    Returns
    -------
    pd.DataFrame
        dataframe with trimmer failure codes

    Raises
    ------
    ValueError
        If the columns are not as expected
    """
    df_trimmer_failure_codes = pd.read_csv(trimmer_failure_codes_csv)
    expected_columns = [
        "read group",
        "code",
        "format",
        "segment",
        "reason",
        "failed read count",
        "total read count",
    ]
    if list(df_trimmer_failure_codes.columns) != expected_columns:
        raise ValueError(
            f"Unexpected columns in {trimmer_failure_codes_csv},"
            f"expected {expected_columns}, got {list(df_trimmer_failure_codes.columns)}"
        )

    # refactor columns and names
    df_trimmer_failure_codes = df_trimmer_failure_codes.rename(
        columns={c: c.replace(" ", "_").lower() for c in df_trimmer_failure_codes.columns}
    )

    # group by segment and reason (aggregate read groups)
    df_trimmer_failure_codes = df_trimmer_failure_codes.groupby(["segment", "reason"]).agg(
        {x: "sum" for x in ("failed_read_count", "total_read_count")}
    )

    # remove segment start if not include_pretrim_filters
    if not include_pretrim_filters and ("start" in df_trimmer_failure_codes.index.get_level_values("segment")):
        pretrim_failed_read_count = df_trimmer_failure_codes[df_trimmer_failure_codes.index.isin(["start"], level=0)][
            "failed_read_count"
        ].sum()
        df_trimmer_failure_codes = df_trimmer_failure_codes.drop(index=["start"], level="segment", errors="ignore")
        df_trimmer_failure_codes["total_read_count"] -= pretrim_failed_read_count
    # remove rsq file if not include_failed_rsq
    elif not include_failed_rsq and (
        "rsq file" in df_trimmer_failure_codes.index.get_level_values("reason")
        or "rsq filter" in df_trimmer_failure_codes.index.get_level_values("reason")
    ):
        rsq_failed_read_count = df_trimmer_failure_codes[
            df_trimmer_failure_codes.index.isin(["rsq file", "rsq filter"], level=1)
        ]["failed_read_count"].sum()
        df_trimmer_failure_codes = df_trimmer_failure_codes.drop(
            index=["rsq file", "rsq filter"], level="reason", errors="ignore"
        )
        df_trimmer_failure_codes["total_read_count"] -= rsq_failed_read_count

    # calculate percentage of failed reads
    df_trimmer_failure_codes = df_trimmer_failure_codes.assign(
        PCT_failure=lambda x: 100 * x["failed_read_count"] / x["total_read_count"]
    )

    # add row with total counts
    if add_total:
        total_row = pd.DataFrame(
            {
                "failed_read_count": df_trimmer_failure_codes["failed_read_count"].sum(),
                "total_read_count": df_trimmer_failure_codes["total_read_count"].iloc[0],
                "PCT_failure": df_trimmer_failure_codes["PCT_failure"].sum(),
            },
            index=pd.MultiIndex.from_tuples([("total", "total")]),
        )

        df_trimmer_failure_codes = pd.concat([df_trimmer_failure_codes, total_row])
        df_trimmer_failure_codes.index = df_trimmer_failure_codes.index.set_names(["segment", "reason"])

    return df_trimmer_failure_codes
