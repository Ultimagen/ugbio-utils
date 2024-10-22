import pandas as pd

BED_COLUMN_CHROM = "chrom"
BED_COLUMN_CHROM_START = "chromStart"
BED_COLUMN_CHROM_END = "chromEnd"


class BedWriter:
    def __init__(self, output_file: str):
        self.fh_var = open(output_file, "w", encoding="utf-8")

    def write(
        self,
        chrom: str,
        start: int,
        end: int,
        description: str = None,
        score: float = None,
    ):
        if start > end:
            raise ValueError(f"start > end in write bed file: {start} > {end}")
        self.fh_var.write(f"{chrom}\t{start}\t{end}")
        if description is not None:
            self.fh_var.write(f"\t{description}")
        if score is not None:
            self.fh_var.write(f"\t{score}")
        self.fh_var.write("\n")

    def close(self):
        self.fh_var.close()


def parse_intervals_file(intervalfile: str, threshold: int = 0, *, sort: bool = True) -> pd.DataFrame:
    """Parses bed file

    Parameters
    ----------
    intervalfile : str
        Input BED file
    threshold : int, optional
        minimal length of interval to output (default = 0)
    sort: bool, optional
        Sort the output dataframe by chromosome and start, optional, default=True

    Returns
    -------
    pd.DataFrame
        Output dataframe with columns chromosome, start, end
    """
    intervals_df = pd.read_csv(
        intervalfile,
        names=["chromosome", "start", "end"],
        usecols=[0, 1, 2],
        index_col=None,
        sep="\t",
    )
    if threshold > 0:
        intervals_df = intervals_df[intervals_df["end"] - intervals_df["start"] > threshold]
    if sort:
        intervals_df = intervals_df.sort_values(["chromosome", "start"])
    intervals_df["chromosome"] = intervals_df["chromosome"].astype("string")
    return intervals_df
