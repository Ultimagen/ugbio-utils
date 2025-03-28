import ast
from collections.abc import Callable

import numpy as np
import pandas as pd

from ugbio_filtering.variant_filtering_utils import VariantSelectionFunctions


class Blacklist:
    """
    Class that stores the blacklist.

    Attributes
    ----------
    blacklist: set
        The blacklist of positions
    annotation: str
        Name of the blacklist
    selection_fcn: Callable
        The function that selects the relevant calls from the variant dataframe

    Parameters
    ---------
    blacklist: set
    annotation: str
    selection_fcn: Callable
    """

    def __init__(self, blacklist: set, annotation: str, selection_fcn: Callable, description: str):
        self.blacklist = blacklist
        self.annotation = annotation
        self.selection_fcn = selection_fcn
        self.description = description

    def apply(self, df: pd.DataFrame) -> pd.Series:
        """Applies the blacklist on the dataframe

        Parameters
        ----------
        df : pd.DataFrame
            Input concordance dataframe

        Returns
        -------
        pd.Series
            Series with string annotation for the blacklist
        """

        select = self.selection_fcn(df)
        idx = set(df[select].index)
        common_with_blacklist = idx & self.blacklist
        result = pd.Series("PASS", index=df.index, dtype=str)
        result.loc[list(common_with_blacklist)] = self.annotation
        return result

    def __str__(self):
        return f"{self.annotation}: {self.description} with {len(self.blacklist)} elements"


def merge_blacklists(blacklists: list) -> pd.Series | None:
    """Combines blacklist annotations from multiple blacklists. Note that the merge
    does not make annotations unique and does not remove PASS from failed annotations

    Parameters
    ----------
    blacklists : list
        list of annotations from blacklist.apply

    Returns
    -------
    pd.Series
        Combined annotations
    """
    if len(blacklists) == 0:
        return None
    if len(blacklists) == 1:
        return blacklists[0]

    concat = blacklists[0].str.cat(blacklists[1:], sep=";", na_rep="PASS")

    return concat


def blacklist_cg_insertions(df: pd.DataFrame) -> pd.Series:
    """
    Removes CG insertions from calls

    Parameters
    ----------
    df: pd.DataFrame
        calls concordance

    Returns
    -------
    pd.Series
    """
    ggc_filter = df["alleles"].apply(lambda x: "GGC" in x or "CCG" in x)
    blank = pd.Series("PASS", dtype=str, index=df.index)
    blank = blank.where(~ggc_filter, "CG_NON_HMER_INDEL")
    return blank


def create_blacklist_statistics_table(df: pd.DataFrame, classify_column: str) -> pd.DataFrame:
    """
    Creates a table in the following format:
    #dbsnp
    #unknown
    #blacklist
    In order to have statistics on how many varints were in each category when we trained.
    @param df: pd.DataFrame
        calls concordance
    @param classify_column:
        Classification column
    @return:
        pd.Series
    """

    return pd.DataFrame(
        [
            np.sum(df[classify_column] == "tp"),
            np.sum(df[classify_column] == "unknown"),
            np.sum(df[classify_column] == "fp"),
        ],
        index=["dbsnp", "unknown", "blacklist"],
        columns=["Categories"],
    )


def load_blacklist_from_bed(bed_path: str, *, with_alleles: bool, description: str = None) -> Blacklist:
    """
    @param bed_path: path to blacklist bed file
    @param with_alleles: whether bed file has alleles column, currently IGNORE it
    @param description: blacklist description
    @return: blacklist object
    """
    if with_alleles:
        exclude_list_df = pd.read_csv(bed_path, sep="\t", names=["chrom", "pos-1", "pos", "alleles"])
        exclude_list_df["alleles"] = exclude_list_df["alleles"].apply(lambda x: np.array(ast.literal_eval(x)))
        exclude_list_df.index = zip(exclude_list_df["chrom"], exclude_list_df["pos"], strict=False)

        blacklist = Blacklist(
            set(exclude_list_df.index),
            annotation="BLACKLIST",
            selection_fcn=VariantSelectionFunctions.ALL,
            description=description,
        )
    else:
        exclude_list_df = pd.read_csv(bed_path, sep="\t", names=["chrom", "pos-1", "pos"])
        exclude_list_df.index = zip(exclude_list_df["chrom"], exclude_list_df["pos_1"], strict=False)
        blacklist = Blacklist(
            set(exclude_list_df.index),
            annotation="BLACKLIST",
            selection_fcn=VariantSelectionFunctions.ALL,
            description=description,
        )
    return blacklist
