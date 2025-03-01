# noqa: N999
import itertools
import json
import os
from collections import defaultdict
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
import seaborn as sns
from ugbio_core.flow_format.flow_based_read import generate_key_from_sequence
from ugbio_core.plotting_utils import set_pyplot_defaults
from ugbio_core.reports.report_utils import generate_report as generate_report_func
from ugbio_core.sorter_utils import (
    plot_read_length_histogram,
    read_and_parse_sorter_statistics_csv,
)
from ugbio_core.trimmer_utils import (
    merge_trimmer_histograms,
    read_trimmer_failure_codes,
)
from ugbio_core.vcfbed.variant_annotation import VcfAnnotator

from ugbio_ppmseq.ppmSeq_consts import STRAND_RATIO_AXIS_LABEL


# Supported adapter versions
class PpmseqAdapterVersions(Enum):
    LEGACY_V5_START = "legacy_v5_start"
    LEGACY_V5 = "legacy_v5"
    LEGACY_V5_END = "legacy_v5_end"
    V1 = "v1"
    DMBL = "dmbl"


# Trimmer segment labels and tags
class TrimmerSegmentLabels(Enum):
    T_HMER_START = "T_hmer_start"
    T_HMER_END = "T_hmer_end"
    A_HMER_START = "A_hmer_start"
    A_HMER_END = "A_hmer_end"
    NATIVE_ADAPTER = "native_adapter"
    NATIVE_ADAPTER_WITH_C = "native_adapter_with_leading_C"
    STEM_END = "Stem_end"  # when native adapter trimming was done on-tool a modified format is used
    START_LOOP = "Start_loop"
    END_LOOP = "End_loop"


class TrimmerSegmentTags(Enum):
    T_HMER_START = "ts"
    T_HMER_END = "te"
    A_HMER_START = "as"
    A_HMER_END = "ae"
    NATIVE_ADAPTER = "a3"
    STEM_END = "s2"  # when native adapter trimming was done on-tool a modified format is used


class TrimmerHistogramSuffixes(Enum):
    NAME = "_name"
    LENGTH = "_length"
    PATTERN_FW = "_pattern_fw"
    MATCH = "_match"


class PpmseqCategories(Enum):
    # Category names
    MIXED = "MIXED"
    MINUS = "MINUS"
    PLUS = "PLUS"
    END_UNREACHED = "END_UNREACHED"
    UNDETERMINED = "UNDETERMINED"


class PpmseqCategoriesConsensus(Enum):
    # Category names
    MIXED = "MIXED"
    MINUS = "MINUS"
    PLUS = "PLUS"
    DISCORDANT = "DISCORDANT"
    UNDETERMINED = "UNDETERMINED"


class HistogramColumnNames(Enum):
    # Internal names
    COUNT = "count"
    COUNT_NORM = "count_norm"
    STRAND_RATIO_START = "strand_ratio_start"
    STRAND_RATIO_END = "strand_ratio_end"
    STRAND_RATIO_CATEGORY_START = "strand_ratio_category_start"
    STRAND_RATIO_CATEGORY_END = "strand_ratio_category_end"
    STRAND_RATIO_CATEGORY_END_NO_UNREACHED = "strand_ratio_category_end_no_unreached"
    STRAND_RATIO_CATEGORY_CONSENSUS = "strand_ratio_category_consensus"
    LOOP_SEQUENCE_START = "loop_sequence_start"
    LOOP_SEQUENCE_END = "loop_sequence_end"
    DUMBBELL_LEFTOVER_START = "Dumbbell_leftover_start"
    ST = "st"  # STRAND_RATIO_CATEGORY_START in ppmSeq V1
    ET = "et"  # STRAND_RATIO_CATEGORY_END in ppmSeq V1


# Input parameter defaults for legacy_v5
STRAND_RATIO_LOWER_THRESH = 0.27
STRAND_RATIO_UPPER_THRESH = 0.73
MIN_TOTAL_HMER_LENGTHS_IN_LOOPS = 4
MAX_TOTAL_HMER_LENGTHS_IN_LOOPS = 8
MIN_STEM_END_MATCHED_LENGTH = 11  # the stem is 12bp, 1 indel allowed as tolerance
DUMBBELL_LEFTOVER_START_MATCH = (
    HistogramColumnNames.DUMBBELL_LEFTOVER_START.value + TrimmerHistogramSuffixes.MATCH.value
)
BASE_PATH = Path(__file__).parent
REPORTS_DIR = "reports"


class PpmseqStrandVcfAnnotator(VcfAnnotator):
    def __init__(
        self,
        adapter_version: str | PpmseqAdapterVersions,
        sr_lower: float = STRAND_RATIO_LOWER_THRESH,
        sr_upper: float = STRAND_RATIO_UPPER_THRESH,
        min_total_hmer_lengths_in_loops: int = MIN_TOTAL_HMER_LENGTHS_IN_LOOPS,
        max_total_hmer_lengths_in_loops: int = MAX_TOTAL_HMER_LENGTHS_IN_LOOPS,
        min_stem_end_matched_length: int = MIN_STEM_END_MATCHED_LENGTH,
    ):
        _assert_adapter_version_supported(adapter_version)
        self.adapter_version = (
            adapter_version.value if isinstance(adapter_version, PpmseqAdapterVersions) else adapter_version
        )
        self.sr_lower = sr_lower
        self.sr_upper = sr_upper
        self.min_total_hmer_lengths_in_loops = min_total_hmer_lengths_in_loops
        self.max_total_hmer_lengths_in_loops = max_total_hmer_lengths_in_loops
        self.min_stem_end_matched_length = min_stem_end_matched_length

    def edit_vcf_header(self, header: pysam.VariantHeader) -> pysam.VariantHeader:
        """
        Edit the VCF header to add strand ratio and strand ratio category INFO fields

        Parameters
        ----------
        header : pysam.VariantHeader
            VCF header

        Returns
        -------
        pysam.VariantHeader
            VCF header with strand ratio and strand ratio category INFO fields

        """
        header.add_line(f"##ppmSeq_adapter_version={self.adapter_version}")
        header.add_line(
            f"##python_cmd:add_strand_ratios_and_categories_to_featuremap=MIXED is {self.sr_lower}-{self.sr_upper}"
        )
        header.info.add(
            id=HistogramColumnNames.STRAND_RATIO_START.value,
            type="Float",
            number=1,
            description="Ratio of MINUS and PLUS strands measured from the tag in the start of the read",
        )
        header.info.add(
            id=HistogramColumnNames.STRAND_RATIO_END.value,
            type="Float",
            number=1,
            description="Ratio of MINUS and PLUS strands measured from the tag in the end of the read",
        )
        header.info.add(
            id=HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value,
            type="String",
            number=1,
            description="ppmSeq read category derived from the ratio of MINUS and PLUS strands "
            "measured from the tag in the start of the read, options: "
            f'{", ".join(ppmseq_category_list)}',
        )
        header.info.add(
            id=HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value,
            type="String",
            number=1,
            description="ppmSeq read category derived from the ratio of MINUS and PLUS strands "
            "measured from the tag in the end of the read, options: "
            f'{", ".join(ppmseq_category_list)}',
        )
        return header

    # TODO: refactor
    def process_records(self, records: list[pysam.VariantRecord]) -> list[pysam.VariantRecord]:  # noqa: C901
        """
        Add strand ratio and strand ratio category INFO fields to the VCF records

        Parameters
        ----------
        records : list[pysam.VariantRecord]
            list of VCF records

        Returns
        -------
        list[pysam.VariantRecord]
            list of VCF records with strand ratio and strand ratio category INFO fields
        """
        records_out = [None] * len(records)
        for j, record in enumerate(records):
            # get the ppmSeq tags from the VCF record
            ppmseq_tags = defaultdict(int)
            for x in [v.value for v in TrimmerSegmentTags.__members__.values()]:
                if x in record.info:
                    ppmseq_tags[x] = int(record.info.get(x))
            # add the start strand ratio and strand ratio category columns to the VCF record
            if self.adapter_version in (
                PpmseqAdapterVersions.LEGACY_V5.value,
                PpmseqAdapterVersions.LEGACY_V5_START.value,
            ):  # legacy_v5_start has start tags only
                # assign to simple variables for readability
                t_hmer_start = ppmseq_tags[TrimmerSegmentTags.T_HMER_START.value]
                a_hmer_start = ppmseq_tags[TrimmerSegmentTags.A_HMER_START.value]
                # determine ratio and category
                tags_sum_start = t_hmer_start + a_hmer_start
                if self.min_total_hmer_lengths_in_loops <= tags_sum_start <= self.max_total_hmer_lengths_in_loops:
                    record.info[HistogramColumnNames.STRAND_RATIO_START.value] = t_hmer_start / (
                        t_hmer_start + a_hmer_start
                    )
                else:
                    record.info[HistogramColumnNames.STRAND_RATIO_START.value] = np.nan
                record.info[HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value] = get_strand_ratio_category(
                    record.info[HistogramColumnNames.STRAND_RATIO_START.value],
                    self.sr_lower,
                    self.sr_upper,
                )
            if self.adapter_version in (
                PpmseqAdapterVersions.LEGACY_V5.value,
                PpmseqAdapterVersions.LEGACY_V5_END.value,
            ):  # legacy_v5_end has end tags only
                # assign to simple variables for readability
                t_hmer_end = ppmseq_tags[TrimmerSegmentTags.T_HMER_END.value]
                a_hmer_end = ppmseq_tags[TrimmerSegmentTags.A_HMER_END.value]
                # determine ratio and category
                tags_sum_end = t_hmer_end + a_hmer_end
                # determine if read end was reached
                is_end_reached = (
                    ppmseq_tags[TrimmerSegmentTags.NATIVE_ADAPTER.value] >= 1
                    or ppmseq_tags[TrimmerSegmentTags.STEM_END.value] >= self.min_stem_end_matched_length
                )
                record.info[HistogramColumnNames.STRAND_RATIO_END.value] = np.nan
                if not is_end_reached:
                    record.info[HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value] = (
                        PpmseqCategories.END_UNREACHED.value
                    )
                else:
                    if self.min_total_hmer_lengths_in_loops <= tags_sum_end <= self.max_total_hmer_lengths_in_loops:
                        record.info[HistogramColumnNames.STRAND_RATIO_END.value] = t_hmer_end / (
                            t_hmer_end + a_hmer_end
                        )
                    record.info[HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value] = get_strand_ratio_category(
                        record.info[HistogramColumnNames.STRAND_RATIO_END.value],
                        self.sr_lower,
                        self.sr_upper,
                    )  # this works for nan values as well - returns UNDETERMINED
            if self.adapter_version in (
                PpmseqAdapterVersions.V1.value,
                PpmseqAdapterVersions.DMBL.value,
            ):  # legacy_v5_start has start tags only
                record.info[HistogramColumnNames.ST.value] = record.info.get(
                    HistogramColumnNames.ST.value, PpmseqCategories.UNDETERMINED.value
                )
                is_end_reached = (
                    ppmseq_tags[TrimmerSegmentTags.NATIVE_ADAPTER.value] >= 1
                    or ppmseq_tags[TrimmerSegmentTags.STEM_END.value] >= self.min_stem_end_matched_length
                )
                record.info[HistogramColumnNames.ET.value] = record.info.get(
                    HistogramColumnNames.ET.value, PpmseqCategories.UNDETERMINED.value
                )
                #  TODO: Check cases where not is_end_reached and et==UNDETERMINED. Make v5 logic conform with this
                if (
                    not is_end_reached
                    and record.info[HistogramColumnNames.ET.value] == PpmseqCategories.UNDETERMINED.value
                ):
                    record.info[HistogramColumnNames.ET.value] = PpmseqCategories.END_UNREACHED.value

            records_out[j] = record

        return records_out


# Misc
ppmseq_category_list = [v.value for v in PpmseqCategories.__members__.values()]
supported_adapter_versions = [e.value for e in PpmseqAdapterVersions]


def _assert_adapter_version_supported(
    adapter_version: str | PpmseqAdapterVersions,
):
    """
    Assert that the adapter version is supported

    Parameters
    ----------
    adapter_version : str | PpmseqAdapterVersions
        adapter version to check

    Raises
    ------
    AssertionError
        If the adapter version is not supported
    """
    if isinstance(adapter_version, PpmseqAdapterVersions):
        if adapter_version.value not in supported_adapter_versions:
            raise AssertionError(
                f"Unsupported adapter version {adapter_version.value}, "
                + f"supported values are {', '.join(supported_adapter_versions)}"
            )
    if isinstance(adapter_version, str):
        if adapter_version not in supported_adapter_versions:
            raise AssertionError(
                f"Unsupported adapter version {adapter_version}, "
                + f"supported values are {', '.join(supported_adapter_versions)}"
            )


def get_strand_ratio_category(strand_ratio, sr_lower, sr_upper) -> str:
    """
    Determine the strand ratio category

    Parameters
    ----------
    strand_ratio : float
        strand ratio
    sr_lower : float
        lower strand ratio threshold for determining strand ratio category MIXED
    sr_upper : float
        upper strand ratio threshold for determining strand ratio category MIXED

    Returns
    -------
    str
        strand ratio category
    """
    if strand_ratio == 0:
        return PpmseqCategories.PLUS.value
    if strand_ratio == 1:
        return PpmseqCategories.MINUS.value
    if sr_lower <= strand_ratio <= sr_upper:
        return PpmseqCategories.MIXED.value
    return PpmseqCategories.UNDETERMINED.value


def read_ppmseq_trimmer_histogram(  # noqa: C901 #TODO: refactor
    adapter_version: str | PpmseqAdapterVersions,
    trimmer_histogram_csv: str,
    sr_lower: float = STRAND_RATIO_LOWER_THRESH,
    sr_upper: float = STRAND_RATIO_UPPER_THRESH,
    min_total_hmer_lengths_in_tags: int = MIN_TOTAL_HMER_LENGTHS_IN_LOOPS,
    max_total_hmer_lengths_in_tags: int = MAX_TOTAL_HMER_LENGTHS_IN_LOOPS,
    min_stem_end_matched_length: int = MIN_STEM_END_MATCHED_LENGTH,
    sample_name: str = "",
    output_filename: str = None,
    *,
    legacy_histogram_column_names: bool = False,
) -> pd.DataFrame:
    """
    Read a ppmSeq trimmer histogram file and add columns for strand ratio and strand ratio category

    Parameters
    ----------
    adapter_version : [str, PpmseqAdapterVersions]
        adapter version to check
    trimmer_histogram_csv : str
        path to a ppmSeq trimmer histogram file
    sr_lower : float, optional
        lower strand ratio threshold for determining strand ratio category
        default 0.27
    sr_upper : float, optional
        upper strand ratio threshold for determining strand ratio category
        default 0.73
    min_total_hmer_lengths_in_tags : int, optional
        minimum total hmer lengths in tags for determining strand ratio category
        default 4
    max_total_hmer_lengths_in_tags : int, optional
        maximum total hmer lengths in tags for determining strand ratio category
        default 8
    min_stem_end_matched_length : int, optional
        minimum length of stem end matched to determine the read end was reached
    sample_name : str, optional
        sample name to use as index, by default ""
    output_filename : str, optional
        path to save dataframe to in parquet format, by default None (not saved).
    legacy_histogram_column_names : bool, optional
        use legacy column names without suffixes, by default False

    Returns
    -------
    pd.DataFrame
        dataframe with strand ratio and strand ratio category columns

    Raises
    ------
    ValueError
        If required columns are missing ("count", "T_hmer_start", "A_hmer_start")
    """
    _assert_adapter_version_supported(adapter_version)
    # read histogram
    df_trimmer_histogram = pd.read_csv(trimmer_histogram_csv)

    # column name suffixes
    name_suffix = TrimmerHistogramSuffixes.NAME.value if not legacy_histogram_column_names else ""
    length_suffix = TrimmerHistogramSuffixes.LENGTH.value if not legacy_histogram_column_names else ""
    pattern_fw_suffix = TrimmerHistogramSuffixes.PATTERN_FW.value if not legacy_histogram_column_names else ""

    # change legacy segment names
    strand_ratio_category_start = HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value
    strand_ratio_category_end = HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value
    loop_sequence_start = HistogramColumnNames.LOOP_SEQUENCE_START.value
    loop_sequence_end = HistogramColumnNames.LOOP_SEQUENCE_END.value
    df_trimmer_histogram = (
        df_trimmer_histogram.rename(
            columns={
                "T hmer": TrimmerSegmentLabels.T_HMER_START.value + length_suffix,
                "A hmer": TrimmerSegmentLabels.A_HMER_START.value + length_suffix,
                "A_hmer_5": TrimmerSegmentLabels.A_HMER_START.value + length_suffix,
                "T_hmer_5": TrimmerSegmentLabels.T_HMER_START.value + length_suffix,
                "A_hmer_3": TrimmerSegmentLabels.A_HMER_END.value + length_suffix,
                "T_hmer_3": TrimmerSegmentLabels.T_HMER_END.value + length_suffix,
                TrimmerSegmentLabels.NATIVE_ADAPTER_WITH_C.value
                + length_suffix: TrimmerSegmentLabels.NATIVE_ADAPTER.value + length_suffix,
            }
        )
        .rename(
            columns={
                f"{TrimmerSegmentLabels.START_LOOP.value}.1": loop_sequence_start,  # Legacy
                f"{TrimmerSegmentLabels.END_LOOP.value}.1": loop_sequence_end,  # Legacy
            }
        )
        .rename(
            columns={
                f"{TrimmerSegmentLabels.START_LOOP.value+name_suffix}": strand_ratio_category_start,
                f"{TrimmerSegmentLabels.END_LOOP.value+name_suffix}": strand_ratio_category_end,
            }
        )
        .rename(
            columns={
                f"{TrimmerSegmentLabels.START_LOOP.value+pattern_fw_suffix}": loop_sequence_start,
                f"{TrimmerSegmentLabels.END_LOOP.value+pattern_fw_suffix}": loop_sequence_end,
            }
        )
    )

    # determine if end was reached - at least 1bp native adapter or all of the end stem were found
    if adapter_version in [
        PpmseqAdapterVersions.LEGACY_V5_END,
        PpmseqAdapterVersions.LEGACY_V5_END.value,
        PpmseqAdapterVersions.LEGACY_V5,
        PpmseqAdapterVersions.LEGACY_V5.value,
        PpmseqAdapterVersions.V1,
        PpmseqAdapterVersions.V1.value,
        PpmseqAdapterVersions.DMBL,
        PpmseqAdapterVersions.DMBL.value,
    ]:
        is_end_reached = (
            df_trimmer_histogram[TrimmerSegmentLabels.NATIVE_ADAPTER.value + length_suffix] >= 1
            if TrimmerSegmentLabels.NATIVE_ADAPTER.value + length_suffix in df_trimmer_histogram.columns
            else df_trimmer_histogram[TrimmerSegmentLabels.STEM_END.value + length_suffix]
            >= min_stem_end_matched_length
        )

    # Handle v5 and v6 loops
    if adapter_version in [
        PpmseqAdapterVersions.LEGACY_V5_START,
        PpmseqAdapterVersions.LEGACY_V5_END,
        PpmseqAdapterVersions.LEGACY_V5,
        PpmseqAdapterVersions.LEGACY_V5_START.value,
        PpmseqAdapterVersions.LEGACY_V5_END.value,
        PpmseqAdapterVersions.LEGACY_V5.value,
    ]:
        # make sure expected columns exist
        for col in (
            HistogramColumnNames.COUNT.value,
            TrimmerSegmentLabels.T_HMER_START.value + length_suffix,
            TrimmerSegmentLabels.A_HMER_START.value + length_suffix,
        ):
            if col not in df_trimmer_histogram.columns:
                raise ValueError(f"Missing expected column {col} in {trimmer_histogram_csv}")
        if (
            TrimmerSegmentLabels.A_HMER_END.value + length_suffix in df_trimmer_histogram.columns
            or TrimmerSegmentLabels.T_HMER_END.value + length_suffix in df_trimmer_histogram.columns
        ) and (
            TrimmerSegmentLabels.NATIVE_ADAPTER.value + length_suffix not in df_trimmer_histogram.columns
            and TrimmerSegmentLabels.STEM_END.value + length_suffix not in df_trimmer_histogram.columns
        ):
            # If an end tag exists (LA-v6)
            raise ValueError(
                f"Missing expected column {TrimmerSegmentLabels.NATIVE_ADAPTER_WITH_C.value + length_suffix} "
                f"or {TrimmerSegmentLabels.NATIVE_ADAPTER.value + length_suffix} "
                f"or {TrimmerSegmentLabels.STEM_END.value + length_suffix} in {trimmer_histogram_csv}"
            )

        df_trimmer_histogram.index.name = sample_name

        # add normalized count column
        df_trimmer_histogram = df_trimmer_histogram.assign(
            count_norm=df_trimmer_histogram[HistogramColumnNames.COUNT.value]
            / df_trimmer_histogram[HistogramColumnNames.COUNT.value].sum()
        )
        # add strand ratio columns and determine categories
        tags_sum_start = (
            df_trimmer_histogram[TrimmerSegmentLabels.T_HMER_START.value + length_suffix]
            + df_trimmer_histogram[TrimmerSegmentLabels.A_HMER_START.value + length_suffix]
        )
        df_trimmer_histogram.loc[:, HistogramColumnNames.STRAND_RATIO_START.value] = (
            (df_trimmer_histogram[TrimmerSegmentLabels.T_HMER_START.value + length_suffix] / tags_sum_start)
            .where(
                (tags_sum_start >= min_total_hmer_lengths_in_tags) & (tags_sum_start <= max_total_hmer_lengths_in_tags)
            )
            .round(2)
        )
        # determine strand ratio category
        df_trimmer_histogram.loc[:, HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value] = df_trimmer_histogram[
            HistogramColumnNames.STRAND_RATIO_START.value
        ].apply(lambda x: get_strand_ratio_category(x, sr_lower, sr_upper))
        if (
            TrimmerSegmentLabels.A_HMER_END.value + length_suffix in df_trimmer_histogram.columns
            or TrimmerSegmentLabels.T_HMER_END.value + length_suffix in df_trimmer_histogram.columns
        ):
            # if only one of the end tags exists (maybe a small subsample) assign the other to 0
            for c in (
                TrimmerSegmentLabels.A_HMER_END.value + length_suffix,
                TrimmerSegmentLabels.T_HMER_END.value + length_suffix,
            ):
                if c not in df_trimmer_histogram.columns:
                    df_trimmer_histogram.loc[:, c] = 0

            tags_sum_end = (
                df_trimmer_histogram[TrimmerSegmentLabels.T_HMER_END.value + length_suffix]
                + df_trimmer_histogram[TrimmerSegmentLabels.A_HMER_END.value + length_suffix]
            )
            df_trimmer_histogram.loc[:, HistogramColumnNames.STRAND_RATIO_END.value] = (
                (
                    df_trimmer_histogram[TrimmerSegmentLabels.T_HMER_END.value + length_suffix]
                    / (
                        df_trimmer_histogram[TrimmerSegmentLabels.T_HMER_END.value + length_suffix]
                        + df_trimmer_histogram[TrimmerSegmentLabels.A_HMER_END.value + length_suffix]
                    )
                )
                .where(
                    (tags_sum_end >= min_total_hmer_lengths_in_tags) & (tags_sum_end <= max_total_hmer_lengths_in_tags)
                )
                .round(2)
            )
            # determine strand ratio category
            df_trimmer_histogram.loc[:, HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value] = (
                df_trimmer_histogram[HistogramColumnNames.STRAND_RATIO_END.value]
                .apply(lambda x: get_strand_ratio_category(x, sr_lower, sr_upper))
                .where(is_end_reached, PpmseqCategories.END_UNREACHED.value)
            )
    # Handle v7 loops
    elif adapter_version in [
        PpmseqAdapterVersions.V1,
        PpmseqAdapterVersions.V1.value,
        PpmseqAdapterVersions.DMBL,
        PpmseqAdapterVersions.DMBL.value,
    ]:
        # In LA-v7 the tags are explicitly detected from the loop sequences
        # an unmatched start tag indicates an undetermined call
        df_trimmer_histogram = df_trimmer_histogram.fillna(
            {HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value: PpmseqCategories.UNDETERMINED.value}
        )
        # determine strand ratio category
        df_trimmer_histogram.loc[:, HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value] = (
            df_trimmer_histogram[HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value]
            .fillna(PpmseqCategories.UNDETERMINED.value)
            .where(is_end_reached, PpmseqCategories.END_UNREACHED.value)
        )
    else:
        raise ValueError(
            f"Unknown adapter version: {adapter_version if isinstance(adapter_version, str) else adapter_version.value}"
        )

    # assign normalized column
    df_trimmer_histogram = df_trimmer_histogram.assign(
        **{
            HistogramColumnNames.COUNT_NORM.value: df_trimmer_histogram[HistogramColumnNames.COUNT.value]
            / df_trimmer_histogram[HistogramColumnNames.COUNT.value].sum()
        }
    )

    # save to parquet
    if output_filename is not None:
        df_trimmer_histogram.to_parquet(output_filename)

    return df_trimmer_histogram


def group_trimmer_histogram_by_strand_ratio_category(
    adapter_version: str | PpmseqAdapterVersions,
    df_trimmer_histogram: pd.DataFrame,
) -> pd.DataFrame:
    """
    Group the trimmer histogram by strand ratio category

    Parameters
    ----------
    adapter_version : str | PpmseqAdapterVersions
        adapter version to check
    df_trimmer_histogram : pd.DataFrame
        dataframe with strand ratio and strand ratio category columns, from read_ppmSeq_strand_trimmer_histogram

    Returns
    -------
    pd.DataFrame
        dataframe with strand ratio category columns as index and strand ratio category columns as columns
    """
    # fill end tag with dummy column if it does not exist
    _assert_adapter_version_supported(adapter_version)
    if HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value not in df_trimmer_histogram.columns:
        df_trimmer_histogram.loc[:, HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value] = np.nan
    # Group by strand ratio category
    count = HistogramColumnNames.COUNT.value
    df_trimmer_histogram_by_category = (
        pd.concat(
            (
                df_trimmer_histogram.groupby(HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value)
                .agg({count: "sum"})
                .rename(columns={count: HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value}),
                df_trimmer_histogram.groupby(HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value)
                .agg({count: "sum"})
                .rename(columns={count: HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value}),
                df_trimmer_histogram.groupby(HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value)
                .agg({count: "sum"})
                .drop(PpmseqCategories.END_UNREACHED.value, errors="ignore")
                .rename(columns={count: HistogramColumnNames.STRAND_RATIO_CATEGORY_END_NO_UNREACHED.value}),
            ),
            axis=1,
        )
        .reindex(ppmseq_category_list)
        .dropna(how="all", axis=1)
        .fillna(0)
        .astype(int)
    )
    # drop dummy column
    if HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value not in df_trimmer_histogram.columns:
        df_trimmer_histogram_by_category = df_trimmer_histogram_by_category.drop(
            [
                HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value,
                HistogramColumnNames.STRAND_RATIO_CATEGORY_END_NO_UNREACHED.value,
            ],
            axis=1,
        )
    return df_trimmer_histogram_by_category


def get_strand_ratio_category_concordance(
    adapter_version: str | PpmseqAdapterVersions,
    df_trimmer_histogram: pd.DataFrame,
) -> pd.DataFrame:
    """
    Get the concordance between the strand ratio categories at the start and end of the read

    Parameters
    ----------
    adapter_version : str | PpmseqAdapterVersions
        adapter version to check
    df_trimmer_histogram : pd.DataFrame
        dataframe with strand ratio and strand ratio category columns, from read_ppmseq_trimmer_histogram

    Returns
    -------
    df_category_concordance: pd.DataFrame
        dataframe with strand ratio category columns as index and strand ratio category columns as columns
    df_category_concordance_no_end_unreached: pd.DataFrame
        dataframe with strand ratio category columns as index and strand ratio category columns as columns, excluding
        reads where the end was unreached


    Raises
    ------
    ValueError
        If the adapter version is not legacy_v5_start and the end tag is missing

    """
    _assert_adapter_version_supported(adapter_version)
    if adapter_version in (
        PpmseqAdapterVersions.LEGACY_V5_START,
        PpmseqAdapterVersions.LEGACY_V5_START.value,
        PpmseqAdapterVersions.LEGACY_V5_END,
        PpmseqAdapterVersions.LEGACY_V5_END.value,
    ):
        raise ValueError(
            f"Adapter version {adapter_version} does not have tags on both ends. "
            "Cannot calculate strand tag category concordance."
        )
    # create concordance dataframe
    df_category_concordance = (
        df_trimmer_histogram.groupby(
            [
                HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value,
                HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value,
            ]
        )
        .agg({HistogramColumnNames.COUNT.value: "sum"})
        .reindex(
            itertools.product(
                [v for v in ppmseq_category_list if v != PpmseqCategories.END_UNREACHED.value],
                ppmseq_category_list,
            )
        )
        .fillna(0)
    )
    df_category_concordance = (df_category_concordance / df_category_concordance.sum().sum()).rename(
        columns={HistogramColumnNames.COUNT.value: HistogramColumnNames.COUNT_NORM.value}
    )
    # create concordance dataframe excluding end unreached reads
    df_category_concordance_no_end_unreached = df_category_concordance.drop(
        PpmseqCategories.END_UNREACHED.value,
        level=HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value,
        errors="ignore",
    )
    df_category_concordance_no_end_unreached = (
        df_category_concordance_no_end_unreached / df_category_concordance_no_end_unreached.sum().sum()
    )
    index_names = df_category_concordance_no_end_unreached.index.names
    df_category_concordance_no_end_unreached = df_category_concordance_no_end_unreached.rename(
        columns={HistogramColumnNames.COUNT.value: HistogramColumnNames.COUNT_NORM.value}
    ).reset_index()
    # create consensus dataframe (combined status of both tags)
    x = HistogramColumnNames.STRAND_RATIO_CATEGORY_CONSENSUS.value  # otherwise flake8 fails on line length
    #   sum consensus categories where both ends are not UNDETERMINED
    df_category_consensus = (
        df_category_concordance_no_end_unreached[
            (
                df_category_concordance_no_end_unreached[HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value]
                == df_category_concordance_no_end_unreached[HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value]
            )
            & (
                df_category_concordance_no_end_unreached[HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value]
                != PpmseqCategories.UNDETERMINED.value
            )
            & (
                df_category_concordance_no_end_unreached[HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value]
                != PpmseqCategories.UNDETERMINED.value
            )
        ]
        .drop(columns=[HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value])
        .rename(columns={HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value: x})
        .set_index(x)
    )
    #   Set UNDETERMINED where at least one is
    df_category_consensus.loc[
        PpmseqCategoriesConsensus.UNDETERMINED.value,
        HistogramColumnNames.COUNT_NORM.value,
    ] = df_category_concordance_no_end_unreached[
        (
            df_category_concordance_no_end_unreached[HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value]
            == PpmseqCategories.UNDETERMINED.value
        )
        | (
            df_category_concordance_no_end_unreached[HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value]
            == PpmseqCategories.UNDETERMINED.value
        )
    ][HistogramColumnNames.COUNT_NORM.value].sum()
    #   the remainder is DISCORDANT - both not UNDETERMINED but do not match
    count = HistogramColumnNames.COUNT_NORM.value  # workaround flake8 contradicts Black and fails on line length
    df_category_consensus.loc[PpmseqCategoriesConsensus.DISCORDANT.value, count] = (
        1 - df_category_consensus[HistogramColumnNames.COUNT_NORM.value].sum()
    )

    return (
        df_category_concordance[HistogramColumnNames.COUNT_NORM.value],
        df_category_concordance_no_end_unreached.set_index(index_names)[HistogramColumnNames.COUNT_NORM.value],
        df_category_consensus[HistogramColumnNames.COUNT_NORM.value],
    )


def read_trimmer_tags_dataframe(
    adapter_version: str | PpmseqAdapterVersions,
    df_strand_ratio_category: str,
    df_category_consensus: str,
    df_sorter_stats: str,
    df_category_concordance: str = None,
) -> pd.DataFrame:
    """
    Read the trimmer tags dataframe

    Parameters
    ----------
    adapter_version : str | PpmseqAdapterVersions
        adapter version to check
    df_strand_ratio_category : pd.DataFrame

    df_category_consensus : pd.DataFrame

    df_sorter_stats : pd.DataFrame

    df_category_concordance : pd.DataFrame, optional

    Returns
    -------
    pd.DataFrame
        dataframe with strand ratio category columns as index and strand ratio category columns as columns

    Raises
    ------
    ValueError
        If the adapter version is not legacy_v5_end and the end tag is missing

    """
    if adapter_version in (
        PpmseqAdapterVersions.LEGACY_V5_START,
        PpmseqAdapterVersions.LEGACY_V5_START.value,
    ):
        df_tags = df_strand_ratio_category.drop(PpmseqCategories.END_UNREACHED.value, errors="ignore")[
            [HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value]
        ]
        df_tags.index = [f"PCT_{x}_reads" for x in df_tags.index]
        df_tags.columns = ["value"]
        df_tags = df_tags * 100 / df_tags.sum()
        df_tags = df_tags["value"]
    elif adapter_version in (
        PpmseqAdapterVersions.LEGACY_V5_END,
        PpmseqAdapterVersions.LEGACY_V5_END.value,
    ):
        df_tags = df_strand_ratio_category[[HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value]]
        df_tags.index = [f"PCT_{x}_reads" for x in df_tags.index]
        df_tags.columns = ["value"]
        df_tags = df_tags * 100 / df_tags.sum()
        df_tags = df_tags["value"]
    elif adapter_version in (
        PpmseqAdapterVersions.LEGACY_V5,
        PpmseqAdapterVersions.LEGACY_V5.value,
        PpmseqAdapterVersions.V1,
        PpmseqAdapterVersions.V1.value,
        PpmseqAdapterVersions.DMBL,
        PpmseqAdapterVersions.DMBL.value,
    ):
        df_tags = df_category_consensus * 100
        undetermined = PpmseqCategoriesConsensus.UNDETERMINED.value
        df_tags = df_tags.rename(
            {
                **{
                    x: f"PCT_{x}_both_tags_where_endreached"
                    for x in (
                        PpmseqCategories.MIXED.value,
                        PpmseqCategories.MINUS.value,
                        PpmseqCategories.PLUS.value,
                    )
                },
                undetermined: f"PCT_{undetermined}_either_tag",
                PpmseqCategoriesConsensus.DISCORDANT.value: f"PCT_{PpmseqCategoriesConsensus.DISCORDANT.value}",
            }
        )
        df_tags.name = "value"
        df_strand_ratio_category_norm = (
            df_strand_ratio_category.loc[
                [PpmseqCategories.END_UNREACHED.value],
                HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value,
            ]
            * 100
            / df_strand_ratio_category[HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value].sum()
        ).T.to_frame()
        df_strand_ratio_category_norm.index = ["PCT_read_end_unreached"]
        df_strand_ratio_category_norm.columns = ["value"]
        df_tags = pd.concat((df_tags, df_strand_ratio_category_norm["value"])).rename(
            {
                HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value: "PCT_read_end_unreached",
            }
        )
        # Calculate mixed reads of total (including end unreached) and mixed read coverage
        mixed_tot = df_category_concordance.loc[(PpmseqCategories.MIXED.value, PpmseqCategories.MIXED.value),]
        df_mixed_cov = pd.DataFrame(
            {
                "MIXED_read_mean_coverage": mixed_tot * df_sorter_stats.loc["Mean_cvg", "value"],
                "PCT_MIXED_both_tags": mixed_tot * 100,
            },
            index=["value"],
        ).T["value"]
        df_tags = pd.concat((df_mixed_cov, df_tags))
    else:
        raise ValueError(
            f"Unknown adapter version: {adapter_version if isinstance(adapter_version, str) else adapter_version.value}"
        )

    df_tags.index.name = "metric"
    return df_tags


def read_trimmer_failure_codes_ppmseq(trimmer_failure_codes_csv: str):
    """
    Read a trimmer failure codes csv file

    Parameters
    ----------
    trimmer_failure_codes_csv : str
        path to a Trimmer failure codes file

    Returns
    -------
    pd.DataFrame
        dataframe with trimmer failure codes

    pd.DataFrame
        dataframe with trimmer application-specific statistics

    Raises
    ------
    AssertionError
        If the columns are not as expected
    """
    df_trimmer_failure_codes = read_trimmer_failure_codes(
        trimmer_failure_codes_csv, include_failed_rsq=False, add_total=True
    )
    adapter_dimers_index = ("insert", "sequence was too short")
    adapter_dimers = (
        df_trimmer_failure_codes.reindex([adapter_dimers_index]).fillna(0).loc[adapter_dimers_index, "PCT_failure"]
    )

    unrecognized_stem_index = ("Stem_start", "no match")
    unrecognized_stem = (
        df_trimmer_failure_codes.reindex([unrecognized_stem_index])
        .fillna(0)
        .loc[unrecognized_stem_index, "PCT_failure"]
    )

    unrecognized_start_loop_index = ("Unrecognized_Start_loop", "sequence was too long")
    unrecognized_start_loop = (
        df_trimmer_failure_codes.reindex([unrecognized_start_loop_index])
        .fillna(0)
        .loc[unrecognized_start_loop_index, "PCT_failure"]
    )

    total_failed_reads_pct = df_trimmer_failure_codes.loc[("total", "total"), "PCT_failure"]

    df_metrics = pd.DataFrame(
        (
            ("PCT_failed_adapter_dimers", adapter_dimers),
            ("PCT_failed_unrecognized_start_stem", unrecognized_stem),
            ("PCT_failed_unrecognized_start_loop", unrecognized_start_loop),
            ("PCT_failed_total", total_failed_reads_pct),
        ),
        columns=["metric", "value"],
    ).set_index("metric")

    return df_trimmer_failure_codes, df_metrics


def read_dumbell_leftover_from_trimmer_histogram(trimmer_histogram_extra_csv, *, legacy_histogram_column_names=False):
    """
    Read the dumbell leftover stats (constant sequences after trimming) from the trimmer histogram

    Parameters
    ----------
    trimmer_histogram_extra_csv : str
        path to a Trimmer histogram extra file
    legacy_histogram_column_names : bool, optional
        use legacy column names without suffixes, by default False

    Returns
    -------
    pd.DataFrame
        dataframe with dumbell leftover stats
    """
    df_dumbbell_leftover = pd.read_csv(trimmer_histogram_extra_csv)
    expected_columns = [DUMBBELL_LEFTOVER_START_MATCH, HistogramColumnNames.COUNT.value]
    if not legacy_histogram_column_names:
        df_dumbbell_leftover = df_dumbbell_leftover.rename(
            columns={c + TrimmerHistogramSuffixes.MATCH.value: c for c in expected_columns}
        )
    if df_dumbbell_leftover.columns.tolist() != expected_columns:
        raise ValueError(f"Unexpected columns {df_dumbbell_leftover.columns.tolist()}, expected {expected_columns}")
    dumbbell_leftover_start_found = "dumbbell_leftover_start_found"
    df_dumbbell_leftover = df_dumbbell_leftover.assign(
        **{dumbbell_leftover_start_found: df_dumbbell_leftover[DUMBBELL_LEFTOVER_START_MATCH].notna()},
    )
    df_dumbbell_leftover = (
        100
        * df_dumbbell_leftover.groupby([dumbbell_leftover_start_found]).agg({HistogramColumnNames.COUNT.value: "sum"})
        / df_dumbbell_leftover[HistogramColumnNames.COUNT.value].sum()
    ).rename(columns={HistogramColumnNames.COUNT.value: "value"})
    df_dumbbell_leftover = (
        df_dumbbell_leftover.reindex([True])
        .fillna(0)
        .assign(
            metric=[
                "PCT_Dumbbell_leftover_in_read_start",
            ]
        )
        .set_index("metric")
    )
    return df_dumbbell_leftover


def collect_statistics(
    adapter_version: str | PpmseqAdapterVersions,
    trimmer_histogram_csv: str,
    sorter_stats_csv: str,
    output_filename: str,
    trimmer_histogram_extra_csv: str = None,
    trimmer_failure_codes_csv: str = None,
    *,
    legacy_histogram_column_names: bool = False,
    **trimmer_histogram_kwargs,
):
    """
    Collect statistics from a ppmSeq trimmer histogram file and a sorter stats file

    Parameters
    ----------
    adapter_version : str | ppmSeqStrandAdapterVersions
        adapter version to check
    trimmer_histogram_csv : str
        path to a ppmSeq Trimmer histogram file
    sorter_stats_csv : str
        path to a Sorter stats file
    output_filename : str
        path to save dataframe to in hdf format (should end with .h5)
    trimmer_histogram_extra_csv : str, optional
        path to a Trimmer histogram extra file, by default None
    trimmer_failure_codes_csv : str, optional
        path to a Trimmer failure codes file, by default None
    legacy_histogram_column_names : bool, optional
        use legacy column names without suffixes, by default False
    trimmer_histogram_kwargs : dict
        additional keyword arguments to pass to read_ppmSeq_strand_trimmer_histogram

    Raises
    ------
    ValueError
        If the adapter version is invalid
    """
    _assert_adapter_version_supported(adapter_version)
    # read Trimmer histogram
    df_trimmer_histogram = read_ppmseq_trimmer_histogram(
        adapter_version,
        trimmer_histogram_csv,
        legacy_histogram_column_names=legacy_histogram_column_names,
        **trimmer_histogram_kwargs,
    )
    df_strand_ratio_category = group_trimmer_histogram_by_strand_ratio_category(adapter_version, df_trimmer_histogram)
    adapter_in_both_ends = adapter_version in (
        PpmseqAdapterVersions.LEGACY_V5,
        PpmseqAdapterVersions.LEGACY_V5.value,
        PpmseqAdapterVersions.V1,
        PpmseqAdapterVersions.V1.value,
        PpmseqAdapterVersions.DMBL,
        PpmseqAdapterVersions.DMBL.value,
    )
    if adapter_in_both_ends:
        df_category_concordance, _, df_category_consensus = get_strand_ratio_category_concordance(
            adapter_version, df_trimmer_histogram
        )
    else:
        df_category_concordance = None
        df_category_consensus = None

    # read Sorter stats
    sorter_stats = read_and_parse_sorter_statistics_csv(sorter_stats_csv)

    # read Trimmer tag stats
    df_tags = read_trimmer_tags_dataframe(
        adapter_version=adapter_version,
        df_strand_ratio_category=df_strand_ratio_category,
        df_category_consensus=df_category_consensus,
        df_sorter_stats=sorter_stats.to_frame(),
        df_category_concordance=df_category_concordance,
    )

    # Merge to create stats shortlist
    df_stats_shortlist = pd.concat((df_tags, sorter_stats))

    # read Trimmer failure tags
    if trimmer_failure_codes_csv:
        df_trimmer_failure_codes, df_failure_codes_metrics = read_trimmer_failure_codes_ppmseq(
            trimmer_failure_codes_csv
        )
        df_stats_shortlist = pd.concat((df_stats_shortlist, df_failure_codes_metrics["value"]))

    is_v7_dumbell = (
        adapter_version
        in (
            PpmseqAdapterVersions.DMBL,
            PpmseqAdapterVersions.DMBL.value,
        )
        and trimmer_histogram_extra_csv
    )
    if is_v7_dumbell:
        df_dumbell_leftover = read_dumbell_leftover_from_trimmer_histogram(
            trimmer_histogram_extra_csv,
            legacy_histogram_column_names=legacy_histogram_column_names,
        )
        df_stats_shortlist = pd.concat((df_stats_shortlist, df_dumbell_leftover))

    # save
    if not output_filename.endswith(".h5"):
        output_filename += ".h5"
    with pd.HDFStore(output_filename, "w") as store:
        keys_to_convert = [
            "stats_shortlist",
            "sorter_stats",
            "strand_ratio_category_counts",
            "strand_ratio_category_norm",
        ]
        store["stats_shortlist"] = df_stats_shortlist
        store["sorter_stats"] = sorter_stats
        store["trimmer_histogram"] = df_trimmer_histogram
        store["strand_ratio_category_counts"] = df_strand_ratio_category
        store["strand_ratio_category_norm"] = df_strand_ratio_category / df_strand_ratio_category.sum()
        if adapter_in_both_ends:
            store["strand_ratio_category_concordance"] = df_category_concordance
            store["strand_ratio_category_consensus"] = df_category_consensus
            keys_to_convert += [
                "strand_ratio_category_concordance",
                "strand_ratio_category_consensus",
            ]
        if trimmer_failure_codes_csv:
            store["trimmer_failure_codes"] = df_trimmer_failure_codes
            keys_to_convert += ["trimmer_failure_codes"]
        if is_v7_dumbell and trimmer_failure_codes_csv:
            store["failure_codes_metrics"] = df_failure_codes_metrics
            keys_to_convert += ["failure_codes_metrics"]
        store["keys_to_convert"] = pd.Series(keys_to_convert)


def add_strand_ratios_and_categories_to_featuremap(
    adapter_version: str | PpmseqAdapterVersions,
    input_featuremap_vcf: str,
    output_featuremap_vcf: str,
    sr_lower: float = STRAND_RATIO_LOWER_THRESH,
    sr_upper: float = STRAND_RATIO_UPPER_THRESH,
    min_total_hmer_lengths_in_tags: int = MIN_TOTAL_HMER_LENGTHS_IN_LOOPS,
    max_total_hmer_lengths_in_tags: int = MAX_TOTAL_HMER_LENGTHS_IN_LOOPS,
    min_stem_end_matched_length: int = MIN_STEM_END_MATCHED_LENGTH,
    chunk_size: int = 10000,
    process_number: int = 1,
):
    """
    Add strand ratio and strand ratio category columns to a featuremap VCF file

    Parameters
    ----------
    adapter_version : str | PpmseqAdapterVersions
        adapter version to check
    input_featuremap_vcf : str
        path to input featuremap VCF file
    output_featuremap_vcf : str
        path to which the output featuremap VCF with the additional fields will be written
    sr_lower : float, optional
        lower strand ratio threshold for determining strand ratio category
        default 0.27
    sr_upper : float, optional
        upper strand ratio threshold for determining strand ratio category
        default 0.73
    min_total_hmer_lengths_in_tags : int, optional
        minimum total hmer lengths in tags for determining strand ratio category
        default 4
    max_total_hmer_lengths_in_tags : int, optional
        maximum total hmer lengths in tags for determining strand ratio category
        default 8
    min_stem_end_matched_length : int, optional
        minimum length of stem end matched to determine the read end was reached
    chunk_size : int, optional
        The chunk size. Defaults to 10000.
    process_number: int, optional
        The number of processes to use. Defaults to 1.
    """
    _assert_adapter_version_supported(adapter_version)
    ppmseq_variant_annotator = PpmseqStrandVcfAnnotator(
        adapter_version=adapter_version,
        sr_lower=sr_lower,
        sr_upper=sr_upper,
        min_total_hmer_lengths_in_loops=min_total_hmer_lengths_in_tags,
        max_total_hmer_lengths_in_loops=max_total_hmer_lengths_in_tags,
        min_stem_end_matched_length=min_stem_end_matched_length,
    )
    PpmseqStrandVcfAnnotator.process_vcf(
        annotators=[ppmseq_variant_annotator],
        input_path=input_featuremap_vcf,
        output_path=output_featuremap_vcf,
        chunk_size=chunk_size,
        process_number=process_number,
    )


def plot_ppmseq_strand_ratio(
    adapter_version: str | PpmseqAdapterVersions,
    df_trimmer_histogram: pd.DataFrame,
    title: str = "",
    output_filename: str = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Plot the strand ratio histogram

    Parameters
    ----------
    adapter_version : str | PpmseqAdapterVersions
        adapter version to check
    df_trimmer_histogram : pd.DataFrame
        dataframe with strand ratio and strand ratio category columns, from read_ppmSeq_strand_trimmer_histogram
    title : str, optional
        plot title, by default ""
    output_filename : str, optional
        path to save the plot to, by default None (not saved)
    ax : matplotlib.axes.Axes, optional
        axes to plot on, by default None (new figure created)

    """
    _assert_adapter_version_supported(adapter_version)
    # display settings
    set_pyplot_defaults()
    if ax is None:
        plt.figure(figsize=(12, 4))
        ax = plt.gca()
    else:
        plt.sca(ax)

    ylim_max = 0
    colors = {
        HistogramColumnNames.STRAND_RATIO_START.value: "xkcd:royal blue",
        HistogramColumnNames.STRAND_RATIO_END.value: "xkcd:red orange",
    }
    markers = {
        HistogramColumnNames.STRAND_RATIO_START.value: "o",
        HistogramColumnNames.STRAND_RATIO_END.value: "s",
    }
    # plot strand ratio histograms for both start and end tags
    for sr, sr_category, sr_category_hist, label in zip(
        (
            HistogramColumnNames.STRAND_RATIO_START.value,
            HistogramColumnNames.STRAND_RATIO_END.value,
        ),
        (
            HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value,
            HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value,
        ),
        (
            HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value,
            HistogramColumnNames.STRAND_RATIO_CATEGORY_END_NO_UNREACHED.value,
        ),
        ("Start tag", "End tag"),
        strict=False,
    ):
        if sr in df_trimmer_histogram.columns:
            # group by strand ratio and strand ratio category for non-undetermined reads
            df_plot = (
                df_trimmer_histogram.sort_values(HistogramColumnNames.COUNT.value, ascending=False)
                .dropna(subset=[sr])
                .groupby(sr)
                .agg({HistogramColumnNames.COUNT_NORM.value: "sum", sr_category: "first"})
                .reset_index()
            )
            # normalize by the total number of reads where the end was reached
            total_reads_norm_count_with_end_reached = df_plot.query(
                f"({sr_category} != '{PpmseqCategories.END_UNREACHED}') and "
                f"({sr_category} != '{PpmseqCategories.UNDETERMINED}')"
            )[HistogramColumnNames.COUNT_NORM.value].sum()
            y = df_plot[HistogramColumnNames.COUNT_NORM.value] / total_reads_norm_count_with_end_reached
            # get category counts
            df_trimmer_histogram_by_strand_ratio_category = group_trimmer_histogram_by_strand_ratio_category(
                adapter_version, df_trimmer_histogram
            )
            mixed_reads_ratio = (
                df_trimmer_histogram_by_strand_ratio_category.loc[PpmseqCategories.MIXED.value, sr_category_hist]
                / df_trimmer_histogram_by_strand_ratio_category[sr_category_hist].sum()
            )
            # plot
            plt.plot(
                df_plot[sr],
                y,
                "-",
                c=colors[sr],
                marker=markers[sr],
                label=f"{label}: {mixed_reads_ratio:.1%}" " mixed reads",
            )
            ylim_max = max(ylim_max, y.max() + 0.07)
    legend_handle = plt.legend(loc="upper left", fontsize=14, fancybox=True, framealpha=0.95)
    title_handle = plt.title(title, fontsize=24)

    plt.xlabel(STRAND_RATIO_AXIS_LABEL)
    plt.ylabel("Relative abundance", fontsize=20)
    plt.ylim(0, ylim_max)
    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )
    return ax


def plot_strand_ratio_category(
    adapter_version: str | PpmseqAdapterVersions,
    df_trimmer_histogram: pd.DataFrame,
    title: str = "",
    output_filename: str = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Plot the strand ratio category histogram

    Parameters
    ----------
    adapter_version : str | PpmseqAdapterVersions
        adapter version to check
    df_trimmer_histogram : pd.DataFrame
        dataframe with strand ratio and strand ratio category columns, from read_ppmseq_trimmer_histogram
    title : str, optional
        plot title, by default ""
    output_filename : str, optional
        path to save the plot to, by default None (not saved)
    ax : matplotlib.axes.Axes, optional
        axes to plot on, by default None (new figure created)

    Returns
    -------
        matplotlib.axes.Axes

    """
    _assert_adapter_version_supported(adapter_version)
    # display settings
    set_pyplot_defaults()
    if ax is None:
        plt.figure(figsize=(14, 4))
        ax = plt.gca()
    else:
        plt.sca(ax)

    # group by category

    df_trimmer_histogram_by_strand_ratio_category = (
        (group_trimmer_histogram_by_strand_ratio_category(adapter_version, df_trimmer_histogram))
        .drop(PpmseqCategories.END_UNREACHED.value, errors="ignore")
        .drop(
            columns=[HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value],
            errors="ignore",
        )
        .rename(
            columns={
                HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value: "Start tag strand ratio category",
                HistogramColumnNames.STRAND_RATIO_CATEGORY_END_NO_UNREACHED.value: "End tag strand ratio category",
            }
        )
    )
    df_plot = (
        (df_trimmer_histogram_by_strand_ratio_category / df_trimmer_histogram_by_strand_ratio_category.sum())
        .reset_index()
        .melt(id_vars="index", var_name="")
    )
    # plot
    sns.barplot(
        data=df_plot,
        x="index",
        y="value",
        hue="",
        ax=ax,
    )
    for cont in ax.containers:
        ax.bar_label(cont, labels=[f"{x:.1%}" for x in cont.datavalues], label_type="edge")
    plt.xticks(rotation=10, ha="center")
    plt.xlabel("")
    plt.ylabel("Relative abundance", fontsize=22)
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], ylim[1] + 0.04)
    legend_handle = plt.legend(bbox_to_anchor=(1.01, 1), fontsize=14, framealpha=0.95)
    title_handle = plt.title(title)
    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle, legend_handle],
        )
    return ax


def plot_strand_ratio_category_concordnace(
    adapter_version: str | PpmseqAdapterVersions,
    df_trimmer_histogram: pd.DataFrame,
    title: str = "",
    output_filename: str = None,
    axs: list[plt.Axes] = None,
) -> list[plt.Axes]:
    """
    Plot the strand ratio category concordance heatmap

    Parameters
    ----------
    adapter_version : str | PpmseqAdapterVersions
        adapter version to check
    df_trimmer_histogram : pd.DataFrame
        dataframe with strand ratio and strand ratio category columns, from read_ppmseq_trimmer_histogram
    title : str, optional
        plot title, by default ""
    output_filename : str, optional
        path to save the plot to, by default None (not saved)
    axs : matplotlib.axes.Axes, optional
        axes to plot on, by default None (new figure created)

    Returns
    -------
    list of axes objects to which the output was plotted

    """
    _assert_adapter_version_supported(adapter_version)
    # get concordance
    df_category_concordance, df_category_concordance_no_end_unreached, _ = get_strand_ratio_category_concordance(
        adapter_version, df_trimmer_histogram
    )

    # display settings
    set_pyplot_defaults()
    if axs is None:
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        fig.subplots_adjust(hspace=0.7)
    plt.suptitle(title)
    # plot
    for ax, subtitle, df_plot in zip(
        axs,
        ("All reads", "Only reads where end was reached"),
        (df_category_concordance, df_category_concordance_no_end_unreached),
        strict=False,
    ):
        df_plot = df_plot.to_frame().unstack().droplevel(0, axis=1)  # noqa: PD010, PLW2901
        df_plot = df_plot.loc[  # noqa: PLW2901
            [v for v in ppmseq_category_list if v != PpmseqCategories.END_UNREACHED.value],
            [v for v in ppmseq_category_list if v in df_plot.columns],
        ].fillna(0)
        df_plot.index.name = "Start tag category"
        df_plot.columns.name = "End tag category"

        if df_plot.shape[0] == 0:
            continue

        sns.heatmap(
            df_plot,
            annot=True,
            fmt=".1%",
            cmap="rocket",
            linewidths=4,
            linecolor="white",
            cbar=False,
            ax=ax,
            annot_kws={"size": 18},
        )
        ax.grid(visible=False)
        plt.sca(ax)
        plt.xticks(rotation=20)
        title_handle = ax.set_title(subtitle, fontsize=20)
    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        plt.savefig(
            output_filename,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle],
        )
    return axs


def plot_trimmer_histogram(  # noqa: C901, PLR0912, PLR0915 #TODO: refactor
    adapter_version: str | PpmseqAdapterVersions,
    df_trimmer_histogram: pd.DataFrame,
    title: str = "",
    output_filename: str = None,
    min_total_hmer_lengths_in_tags: int = MIN_TOTAL_HMER_LENGTHS_IN_LOOPS,
    max_total_hmer_lengths_in_tags: int = MAX_TOTAL_HMER_LENGTHS_IN_LOOPS,
    *,
    legacy_histogram_column_names: bool = False,
) -> list[plt.Axes]:
    """
    Plot the trimmer hmer calls on a heatmap

    Parameters
    ----------
    adapter_version : str | PpmseqAdapterVersions
        adapter version to check
    df_trimmer_histogram : pd.DataFrame
        dataframe with strand ratio and strand ratio category columns, from read_ppmseq_trimmer_histogram
    title : str, optional
        plot title, by default ""
    output_filename : str, optional
        path to save the plot to, by default None (not saved)
    axs : matplotlib.axes.Axes, optional
        axes to plot on, by default None (new figure created)
    min_total_hmer_lengths_in_tags : int, optional
        minimum total hmer lengths in tags for determining strand ratio category
        default 4
    max_total_hmer_lengths_in_tags : int, optional
        maximum total hmer lengths in tags for determining strand ratio category
        default 8
    legacy_histogram_column_names : bool, optional
        use legacy column names without suffixes, by default False

    Returns
    -------
    axs: list[plt.Axes]
        list of axes objects to which the output was plotted

    Raises
    ------
    ValueError
        If the adapter version is invalid

    """
    _assert_adapter_version_supported(adapter_version)
    # display settings
    set_pyplot_defaults()

    # column name suffixes
    length_suffix = TrimmerHistogramSuffixes.LENGTH.value if not legacy_histogram_column_names else ""

    if adapter_version in (
        PpmseqAdapterVersions.LEGACY_V5,
        PpmseqAdapterVersions.LEGACY_V5.value,
        PpmseqAdapterVersions.LEGACY_V5_START,
        PpmseqAdapterVersions.LEGACY_V5_START.value,
        PpmseqAdapterVersions.LEGACY_V5_END,
        PpmseqAdapterVersions.LEGACY_V5_END.value,
    ):
        # generate axs
        if adapter_version in (
            PpmseqAdapterVersions.LEGACY_V5,
            PpmseqAdapterVersions.LEGACY_V5.value,
        ):
            fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True, sharex=True)
            fig.subplots_adjust(wspace=0.3)
            plot_iter = (
                (
                    TrimmerSegmentLabels.A_HMER_START.value + length_suffix,
                    TrimmerSegmentLabels.T_HMER_START.value + length_suffix,
                    HistogramColumnNames.COUNT_NORM.value,
                    "Start loop",
                ),
                (
                    TrimmerSegmentLabels.A_HMER_END.value + length_suffix,
                    TrimmerSegmentLabels.T_HMER_END.value + length_suffix,
                    HistogramColumnNames.COUNT_NORM.value,
                    "End loop",
                ),
            )
        else:
            fig, axs = plt.subplots(1, 1, figsize=(8, 6))
            axs = [axs]
            if adapter_version in (
                PpmseqAdapterVersions.LEGACY_V5_START,
                PpmseqAdapterVersions.LEGACY_V5_START.value,
            ):
                plot_iter = (
                    (
                        TrimmerSegmentLabels.A_HMER_START.value + length_suffix,
                        TrimmerSegmentLabels.T_HMER_START.value + length_suffix,
                        HistogramColumnNames.COUNT_NORM.value,
                        "Start loop",
                    ),
                )
            elif adapter_version in (
                PpmseqAdapterVersions.LEGACY_V5_END,
                PpmseqAdapterVersions.LEGACY_V5_END.value,
            ):
                plot_iter = (
                    (
                        TrimmerSegmentLabels.A_HMER_END.value + length_suffix,
                        TrimmerSegmentLabels.T_HMER_END.value + length_suffix,
                        HistogramColumnNames.COUNT_NORM.value,
                        "End loop",
                    ),
                )

        # plot
        title_handle = plt.suptitle(title, y=1.03)
        for ax, (xcol, ycol, zcol, subtitle) in zip(axs, plot_iter, strict=False):
            # group by strand ratio and strand ratio category for non-undetermined reads
            df_plot = df_trimmer_histogram.groupby([xcol, ycol]).agg({zcol: "sum"}).reset_index()
            df_hmer_sum = df_plot[[xcol, ycol]].sum(axis=1)
            df_plot = df_plot[
                (min_total_hmer_lengths_in_tags <= df_hmer_sum) & (df_hmer_sum <= max_total_hmer_lengths_in_tags)
            ]
            df_plot.loc[:, zcol] = df_plot[zcol] / df_plot[zcol].sum()
            # plot
            plt.sca(ax)
            plt.scatter(df_plot[xcol], df_plot[ycol], s=500 * df_plot[zcol], c=df_plot[zcol])
            plt.colorbar()
            plt.xticks(range(int(plt.gca().get_xlim()[1]) + 1))
            plt.yticks(range(int(plt.gca().get_ylim()[1]) + 1))
            plt.xlabel(xcol.replace("_", " "))
            plt.ylabel(ycol.replace("_", " "))
            plt.title(subtitle, fontsize=22)
    elif adapter_version in (
        PpmseqAdapterVersions.V1,
        PpmseqAdapterVersions.V1.value,
        PpmseqAdapterVersions.DMBL,
        PpmseqAdapterVersions.DMBL.value,
    ):
        fig, axs_all_both = plt.subplots(3, 10, figsize=(18, 5), sharex=False, sharey=True)
        fig.subplots_adjust(wspace=0.25, hspace=0.6)
        title_handle = fig.suptitle(title, y=1.25)
        for ax in axs_all_both.flatten():
            ax.grid(visible=False)
        for ax in axs_all_both[:, 4:6].flatten():
            ax.axis("off")
        axs_all_both[0, 4].text(0.5, 1.2, "Expected\nsignal", fontsize=20)
        axs_all_both[0, 4].text(0.5, 0.5, "1  1  1  1", fontsize=18)
        axs_all_both[1, 4].text(0.5, 0.5, "0  2  0  2", fontsize=18)
        axs_all_both[2, 4].text(0.5, 0.5, "2  0  2  0", fontsize=18)
        axs_all_both[0, 1].text(0, 2, "Start loop", fontsize=28)
        axs_all_both[0, 6].text(0, 2, "End loop", fontsize=28)
        flow_order_start = "ATGC"
        flow_order_end = "GCAT"
        for m, (loop, loop_category, flow_order, append_base) in enumerate(
            zip(
                (
                    HistogramColumnNames.LOOP_SEQUENCE_START.value,
                    HistogramColumnNames.LOOP_SEQUENCE_END.value,
                ),
                (
                    HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value,
                    HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value,
                ),
                (flow_order_start, flow_order_end),
                ("", "C"),
                strict=False,
            )
        ):
            df_calls = (
                df_trimmer_histogram.groupby(
                    [
                        loop_category,
                        loop,
                    ]
                )
                .agg({HistogramColumnNames.COUNT_NORM.value: "sum"})
                .query(f"{HistogramColumnNames.COUNT_NORM.value} > 0.0001")
            )
            axs_all = axs_all_both[:, m * 6 : m * 6 + 4]

            for k, (cat, expected_signal) in enumerate(
                zip(
                    (
                        PpmseqCategories.MIXED.value,
                        PpmseqCategories.MINUS.value,
                        PpmseqCategories.PLUS.value,
                    ),
                    (
                        [1, 1, 1, 1],
                        [0, 2, 0, 2],
                        [2, 0, 2, 0],
                    ),
                    strict=False,
                )
            ):
                if cat not in df_calls.index.get_level_values(loop_category):
                    continue
                df_calls_x = df_calls.loc[cat].reset_index()
                # generate flow signal, append a C to the end of the sequence for the case where that C was trimmed
                # due to a wrong Trimmer format, so this code would still work when the "loop sequence" does not
                # contain the C
                df_calls_x = df_calls_x.assign(
                    flow_signal=df_calls_x[loop].apply(
                        lambda x, flow_order=flow_order, append_base=append_base: generate_key_from_sequence(
                            x + append_base, flow_order=flow_order
                        )
                    )
                )
                df_calls_x = (
                    df_calls_x["flow_signal"]
                    .apply(pd.Series)
                    .assign(
                        **{HistogramColumnNames.COUNT_NORM.value: df_calls_x[HistogramColumnNames.COUNT_NORM.value]}
                    )
                )
                axs = axs_all[k, :]
                axs[0].set_ylabel(cat, fontsize=20)
                for ax, (j, base) in zip(axs, enumerate(flow_order), strict=False):
                    x = (
                        df_calls_x.groupby(j)
                        .agg({HistogramColumnNames.COUNT_NORM.value: "sum"})
                        .reindex(range(5))
                        .fillna(0)
                        .reset_index()
                    )
                    ax.bar(
                        x[j],
                        x[HistogramColumnNames.COUNT_NORM.value] / x[HistogramColumnNames.COUNT_NORM.value].sum(),
                        color=[
                            "xkcd:red",
                            "xkcd:green",
                            "xkcd:blue",
                            "xkcd:teal",
                            "xkcd:violet",
                        ],
                    )
                    ax.bar(
                        expected_signal[j],
                        1,
                        facecolor="none",
                        edgecolor="k",
                        alpha=1,
                        linewidth=2,
                    )
                    ax.set_xticks(range(5))
                    if k == 0:
                        ax.set_title(base)
                    if k == 2:  # noqa: PLR2004
                        ax.set_xlabel("hmer")
    else:
        raise ValueError(f"Invalid adapter version {adapter_version}")

    if output_filename is not None:
        if not output_filename.endswith(".png"):
            output_filename += ".png"
        fig.savefig(
            output_filename,
            facecolor="w",
            dpi=300,
            bbox_inches="tight",
            bbox_extra_artists=[title_handle],
        )
    return axs


def flatten_strand_ratio_category(h5_file: str, key: str) -> pd.DataFrame:
    """ "
    convert the 2D dataframe of strand ratio categories to a 1D dataframe that is readable by Papyrus
    """
    df = pd.read_hdf(h5_file, key)  # noqa: PD901
    df["index_col"] = df.index
    df = df.reset_index(drop=True)  # noqa: PD901
    df1 = df.melt(id_vars=["index_col"])
    df1[""] = df1["index_col"] + "_" + df1["variable"]
    df2 = df1[["", "value"]]
    df2 = df2.set_index("")
    df2 = df2.T
    df2 = df2.reset_index(drop=True)
    return df2


def flatten_metrics(h5_file: str, key: str) -> pd.DataFrame:
    """
    Convert a 1D dataframe to a 1D dataframe that is readable by Papyrus
    """
    df = pd.read_hdf(h5_file, key)  # noqa: PD901
    df.index.names = [""]
    df = df.T  # noqa: PD901
    df = df.reset_index(drop=True)  # noqa: PD901
    return df


def convert_h5_to_papyrus_json(h5_file: str, output_json: str) -> str:
    """
    Convert a statistics HDF5 file to a Papyrus JSON file
    """
    with pd.HDFStore(h5_file, "r") as store:
        h5_file_keys = store.keys()

    # flatten strand ratio categories
    strand_ratio_to_convert = [
        "strand_ratio_category_counts",
        "strand_ratio_category_norm",
    ]
    flatten_dfs = {
        key: flatten_strand_ratio_category(h5_file, key) for key in strand_ratio_to_convert if f"/{key}" in h5_file_keys
    }

    # flatten 1D dataframes
    keys_to_convert = [
        "stats_shortlist",
        "sorter_stats",
        "strand_ratio_category_consensus",
    ]
    flatten_dfs.update({key: flatten_metrics(h5_file, key) for key in keys_to_convert if f"/{key}" in h5_file_keys})

    # category concordance
    if "/strand_ratio_category_concordance" in h5_file_keys or "strand_ratio_category_concordance" in h5_file_keys:
        df_category_concordance = pd.read_hdf(h5_file, "/strand_ratio_category_concordance")
        df_category_concordance.index = df_category_concordance.index.to_flat_index()
        df_category_concordance = df_category_concordance.to_frame().rename(
            columns={HistogramColumnNames.COUNT_NORM.value: 0}
        )
        df_category_concordance = df_category_concordance.T
        flatten_dfs["category_concordance"] = df_category_concordance

    # wrap all in one json
    root_element = "metrics"
    new_json_dict = {root_element: {}}
    new_json_dict[root_element] = {key: json.loads(df.to_json(orient="table")) for key, df in flatten_dfs.items()}

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(new_json_dict, f, indent=2)


def ppmseq_qc_analysis(  # noqa: C901, PLR0912, PLR0913, PLR0915 #TODO: refactor
    adapter_version: str | PpmseqAdapterVersions,
    trimmer_histogram_csv: list[str],
    sorter_stats_csv: str,
    output_path: str,
    output_basename: str = None,
    sorter_stats_json: str = None,
    trimmer_histogram_extra_csv: list[str] = None,
    trimmer_failure_codes_csv: str = None,
    collect_statistics_kwargs: dict = None,
    generate_report: bool = True,  # noqa: FBT001, FBT002
    keep_temp_visualization_files: bool = False,  # noqa: FBT001, FBT002
    sr_lower: float = STRAND_RATIO_LOWER_THRESH,
    sr_upper: float = STRAND_RATIO_UPPER_THRESH,
    min_total_hmer_lengths_in_tags: int = MIN_TOTAL_HMER_LENGTHS_IN_LOOPS,
    max_total_hmer_lengths_in_tags: int = MAX_TOTAL_HMER_LENGTHS_IN_LOOPS,
    min_stem_end_matched_length: int = MIN_STEM_END_MATCHED_LENGTH,
    legacy_histogram_column_names: bool = False,  # noqa: FBT001, FBT002
    qc_filename_suffix: str = ".ppmSeq.applicationQC",
):
    """
    Run the ppmSeq QC analysis pipeline

    Parameters
    ----------
    adapter_version : str | PpmseqAdapterVersions
        adapter version to check
    trimmer_histogram_csv : list[str]
        path to a ppmSeq Trimmer histogram file
    sorter_stats_csv : str
        path to a Sorter stats file
    output_path : str
        path (folder) to which data and report will be written to
    output_basename : str, optional
        basename for output files, by default None (basename of trimmer_histogram_csv)
    sorter_stats_json : str, optional
        path to a Sorter stats JSON file, by default None
    trimmer_histogram_extra_csv : str, optional
        path to a ppmSeq Trimmer histogram extra file, by default None
    trimmer_failure_codes_csv : str, optional
        path to a ppmSeq Trimmer failure codes file, by default None
    collect_statistics_kwargs : dict, optional
        kwargs for collect_statistics, by default None
    generate_report
        if True, generate an html+jupyter report, by default True
    keep_temp_png_files
        if True, keep temporary png files, by default False
    sr_lower : float, optional
        lower strand ratio threshold for determining strand ratio category
        default 0.27
    sr_upper : float, optional
        upper strand ratio threshold for determining strand ratio category
        default 0.73
    min_total_hmer_lengths_in_tags : int, optional
        minimum total hmer lengths in tags for determining strand ratio category
        default 4
    max_total_hmer_lengths_in_tags : int, optional
        maximum total hmer lengths in tags for determining strand ratio category
        default 8
    min_stem_end_matched_length : int, optional
        minimum length of stem end matched to determine the read end was reached
    legacy_histogram_column_names : bool, optional
        use legacy column names without suffixes, by default False
    qc_filename_suffix : str, optional
        suffix for the output statistics file immediately after output_basename, by default ".ppmSeq.applicationQC"

    """
    # Handle input and output files
    # check inputs
    _assert_adapter_version_supported(adapter_version)
    for file in trimmer_histogram_csv:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"{file} not found")

    if not os.path.isfile(sorter_stats_csv):
        raise FileNotFoundError(f"{sorter_stats_csv} not found")

    # make output directory and determine base file name
    os.makedirs(output_path, exist_ok=True)
    if output_basename is None:
        output_basename = os.path.basename(trimmer_histogram_csv[0])
    # main outputs
    output_statistics_h5 = os.path.join(output_path, f"{output_basename}{qc_filename_suffix}.h5")
    output_statistics_json = os.path.join(output_path, f"{output_basename}{qc_filename_suffix}.json")
    output_report_html = Path(output_path) / f"{output_basename}{qc_filename_suffix}.html"
    # Temporary image files
    output_report_ipynb = os.path.join(output_path, f"{output_basename}{qc_filename_suffix}.ipynb")
    output_trimmer_histogram_plot = os.path.join(output_path, f"{output_basename}.trimmer_histogram.png")
    output_strand_ratio_plot = os.path.join(output_path, f"{output_basename}.strand_ratio.png")
    output_strand_ratio_category_plot = os.path.join(output_path, f"{output_basename}.strand_ratio_category.png")
    output_strand_ratio_category_concordance_plot = os.path.join(
        output_path, f"{output_basename}.strand_ratio_category_concordance.png"
    )
    output_read_length_histogram_plot = os.path.join(output_path, f"{output_basename}.read_length_histogram.png")
    output_visualization_files = [
        output_report_ipynb,
        output_trimmer_histogram_plot,
        output_strand_ratio_plot,
        output_strand_ratio_category_plot,
        output_strand_ratio_category_concordance_plot,
        output_read_length_histogram_plot,
    ]

    # Merge Trimmer histograms from different optical paths (APL mode)
    merged_histogram_csv = merge_trimmer_histograms(trimmer_histogram_csv, output_path=output_path)
    merged_histogram_extra_csv = (
        merge_trimmer_histograms(trimmer_histogram_extra_csv, output_path=output_path)
        if trimmer_histogram_extra_csv
        else None
    )

    # collect statistics
    # create the input for collect statistics
    if collect_statistics_kwargs is None:
        collect_statistics_kwargs = {}
    collect_statistics_kwargs.setdefault("output_filename", output_statistics_h5)
    collect_statistics_kwargs.setdefault("adapter_version", adapter_version)
    collect_statistics_kwargs.setdefault("trimmer_histogram_csv", merged_histogram_csv)
    collect_statistics_kwargs.setdefault("sorter_stats_csv", sorter_stats_csv)
    collect_statistics_kwargs.setdefault("trimmer_failure_codes_csv", trimmer_failure_codes_csv)
    collect_statistics_kwargs.setdefault("trimmer_histogram_extra_csv", merged_histogram_extra_csv)
    collect_statistics_kwargs.setdefault("sr_lower", sr_lower)
    collect_statistics_kwargs.setdefault("sr_upper", sr_upper)
    collect_statistics_kwargs.setdefault("min_total_hmer_lengths_in_tags", min_total_hmer_lengths_in_tags)
    collect_statistics_kwargs.setdefault("max_total_hmer_lengths_in_tags", max_total_hmer_lengths_in_tags)
    collect_statistics_kwargs.setdefault("min_stem_end_matched_length", min_stem_end_matched_length)
    collect_statistics_kwargs.setdefault("legacy_histogram_column_names", legacy_histogram_column_names)
    collect_statistics(**collect_statistics_kwargs)

    # read Trimmer histogram output from collect statistics
    df_trimmer_histogram = pd.read_hdf(output_statistics_h5, "trimmer_histogram")

    # convert h5 to Papyrus json
    convert_h5_to_papyrus_json(output_statistics_h5, output_statistics_json)

    # generate plots
    plot_trimmer_histogram(
        adapter_version,
        df_trimmer_histogram,
        title=f"{output_basename} hmer calls",
        output_filename=output_trimmer_histogram_plot,
        legacy_histogram_column_names=legacy_histogram_column_names,
    )
    if adapter_version in [
        PpmseqAdapterVersions.LEGACY_V5_START,
        PpmseqAdapterVersions.LEGACY_V5_END,
        PpmseqAdapterVersions.LEGACY_V5,
        PpmseqAdapterVersions.LEGACY_V5_START.value,
        PpmseqAdapterVersions.LEGACY_V5_END.value,
        PpmseqAdapterVersions.LEGACY_V5.value,
    ]:  # not possible in v7
        plot_ppmseq_strand_ratio(
            adapter_version,
            df_trimmer_histogram,
            title=f"{output_basename} strand ratio",
            output_filename=output_strand_ratio_plot,
        )
    plot_strand_ratio_category(
        adapter_version,
        df_trimmer_histogram,
        title=f"{output_basename} strand ratio category",
        output_filename=output_strand_ratio_category_plot,
    )
    if adapter_version in (
        PpmseqAdapterVersions.LEGACY_V5,
        PpmseqAdapterVersions.LEGACY_V5.value,
        PpmseqAdapterVersions.V1,
        PpmseqAdapterVersions.V1.value,
        PpmseqAdapterVersions.DMBL,
        PpmseqAdapterVersions.DMBL.value,
    ):
        plot_strand_ratio_category_concordnace(
            adapter_version,
            df_trimmer_histogram,
            title=f"{output_basename} strand ratio category concordance",
            output_filename=output_strand_ratio_category_concordance_plot,
        )

    if sorter_stats_json:
        plot_read_length_histogram(
            sorter_stats_json,
            title=f"{output_basename}",
            output_filename=output_read_length_histogram_plot,
        )

    # generate report
    if generate_report:
        template_notebook = BASE_PATH / REPORTS_DIR / "ppmSeq_qc_report.ipynb"
        illustration_file = (
            "ppmSeq_legacy_v5_illustration.png"
            if adapter_version
            in (
                PpmseqAdapterVersions.LEGACY_V5_START.value,
                PpmseqAdapterVersions.LEGACY_V5.value,
                PpmseqAdapterVersions.LEGACY_V5_END.value,
            )
            else "reports/ppmSeq_v1_illustration.png"
        )
        illustration_path = BASE_PATH / REPORTS_DIR / illustration_file
        parameters = {
            "adapter_version": (adapter_version if isinstance(adapter_version, str) else adapter_version.value),
            "statistics_h5": output_statistics_h5,
            "trimmer_histogram_png": output_trimmer_histogram_plot,
            "strand_ratio_category_png": output_strand_ratio_category_plot,
            "sr_lower": sr_lower,
            "sr_upper": sr_upper,
            "min_total_hmer_lengths_in_tags": min_total_hmer_lengths_in_tags,
            "max_total_hmer_lengths_in_tags": max_total_hmer_lengths_in_tags,
            "illustration_file": illustration_path,
            "trimmer_histogram_extra_csv": trimmer_histogram_extra_csv,
            "trimmer_failure_codes_csv": trimmer_failure_codes_csv,
        }
        if adapter_version in (
            PpmseqAdapterVersions.LEGACY_V5,
            PpmseqAdapterVersions.LEGACY_V5.value,
            PpmseqAdapterVersions.LEGACY_V5_START,
            PpmseqAdapterVersions.LEGACY_V5_START.value,
            PpmseqAdapterVersions.LEGACY_V5_END,
            PpmseqAdapterVersions.LEGACY_V5_END.value,
        ):  # not available in v7
            parameters["strand_ratio_png"] = output_strand_ratio_plot
        if adapter_version in (
            PpmseqAdapterVersions.LEGACY_V5,
            PpmseqAdapterVersions.LEGACY_V5.value,
            PpmseqAdapterVersions.V1,
            PpmseqAdapterVersions.V1.value,
            PpmseqAdapterVersions.DMBL,
            PpmseqAdapterVersions.DMBL.value,
        ):
            parameters["strand_ratio_category_concordance_png"] = output_strand_ratio_category_concordance_plot
        if sorter_stats_json:
            parameters["output_read_length_histogram_plot"] = output_read_length_histogram_plot

        # collect temporary png and ipynb files
        if not keep_temp_visualization_files:
            tmp_files = [Path(file) for file in output_visualization_files]
        else:
            tmp_files = None

        # create the html report
        generate_report_func(
            template_notebook_path=template_notebook,
            parameters=parameters,
            output_report_html_path=output_report_html,
            tmp_files=tmp_files,
        )
