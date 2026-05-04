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
from ugbio_core.plotting_utils import set_pyplot_defaults
from ugbio_core.reports.report_utils import generate_report as generate_report_func
from ugbio_core.trimmer_utils import (
    read_trimmer_failure_codes,
)
from ugbio_core.vcfbed.variant_annotation import VcfAnnotator

from ugbio_ppmseq.ppmSeq_consts import STRAND_RATIO_AXIS_LABEL


# Supported adapter versions
class PpmseqAdapterVersions(Enum):
    LEGACY_V5 = "legacy_v5"
    V1 = "v1"


# Trimmer segment labels and tags
class TrimmerSegmentTags(Enum):
    T_HMER_START = "ts"
    T_HMER_END = "te"
    A_HMER_START = "as"
    A_HMER_END = "ae"
    NATIVE_ADAPTER = "a3"
    STEM_END = "s2"  # when native adapter trimming was done on-tool a modified format is used


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
    ST = "st"  # STRAND_RATIO_CATEGORY_START in ppmSeq V1
    ET = "et"  # STRAND_RATIO_CATEGORY_END in ppmSeq V1


# Per-read ppmSeq tag names expected in the subsampled SAM.
SR_TAG = "sr"  # strand ratio (float, from raw signal)
ST_TAG = "st"  # start-loop category (MINUS / PLUS / MIXED / UNDETERMINED)
ET_TAG = "et"  # end-loop category
TM_TAG = "tm"  # trimming reasons; contains "A" when the adapter was seen (end reached)

# Subset of sorter-stats metrics to surface in the report's Table 4 "Sorter stats" section.
# The raw CSV has ~40 rows; Ken asked for just the ones below. Missing keys are silently
# skipped so older sorter versions / partial CSVs still work.
SORTER_STATS_KEYS_TO_SHOW = (
    "PF_Barcode_reads",
    "PCT_PF_Reads_aligned",
    "PCT_PF_Q30_bases",
    "PCT_SOFTCLIPPED_bases",
    "Mean_Read_Length",
    "Mean_cvg",
    "PCT_Quality_Trimmed_Reads",
    "PCT_Adapter_Reached",
    "PCT_Chimeras",
    "Indel_Rate",
)

# Input parameter defaults for the legacy featuremap annotator path
STRAND_RATIO_LOWER_THRESH = 0.27
STRAND_RATIO_UPPER_THRESH = 0.73
MIN_TOTAL_HMER_LENGTHS_IN_LOOPS = 4
MAX_TOTAL_HMER_LENGTHS_IN_LOOPS = 8
MIN_STEM_END_MATCHED_LENGTH = 11  # the stem is 12bp, 1 indel allowed as tolerance
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
            if self.adapter_version == PpmseqAdapterVersions.LEGACY_V5.value:
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
            else:  # v1 — st/et tags already carry the category
                record.info[HistogramColumnNames.ST.value] = record.info.get(
                    HistogramColumnNames.ST.value, PpmseqCategories.UNDETERMINED.value
                )
                record.info[HistogramColumnNames.ET.value] = record.info.get(
                    HistogramColumnNames.ET.value, PpmseqCategories.END_UNREACHED.value
                )

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


def read_tags_from_subsampled_sam(
    sam_path: str,
    sample_name: str = "",
    output_filename: str = None,
) -> pd.DataFrame:
    """
    Read per-read ppmSeq tags (sr, st, et, tm) from the subsampled SAM/BAM/CRAM produced by sorter.

    Parameters
    ----------
    sam_path : str
        Path to the subsampled SAM/BAM/CRAM emitted by sorter when demux was called with
        ``--sample-nr-reads=N``. Pysam autodetects the format from the extension.
    sample_name : str, optional
        Sample name to use as the dataframe index name.
    output_filename : str, optional
        If provided, save the dataframe as parquet.

    Returns
    -------
    pd.DataFrame
        One row per read with columns:
          - ``sr`` (float): strand ratio from raw signal. NaN on reads whose sr tag is
            absent; the report's sr-dependent sections are skipped when every read lacks it.
          - ``st`` (str): start-loop category (MINUS / PLUS / MIXED / UNDETERMINED).
          - ``et`` (str): end-loop category; empty string when the ``et`` tag is absent.
          - ``tm`` (str): trimming-reason string (e.g. ``A``, ``AQZ``). Empty string when
            absent, which means the adapter was not reached — mapped to END_UNREACHED below.
          - ``count``: always 1 — lets downstream code that aggregates on it work unchanged.
          - ``strand_ratio_category_start`` / ``strand_ratio_category_end``:
            copies of ``st`` / ``et`` with END_UNREACHED substituted on rows where ``tm`` has no ``A``.

    Raises
    ------
    KeyError
        If any read is missing the ``st`` tag. The ``st`` tag is required on every ppmSeq
        read produced by the current pipeline; a missing one is a fixture/pipeline bug.
    """
    rows = []
    for_sr_nan = float("nan")
    with pysam.AlignmentFile(sam_path, check_sq=False) as fh:
        for rec in fh:
            # Skip reads that Trimmer couldn't match. Sorter tags them with RG="unmatched"
            # (configurable via TrimmerParameters.failure_read_group, but "unmatched" is the
            # standard value used by our WDL templates). These reads never received ppmSeq
            # tag calls so including them would either raise KeyError on st or skew Section 1
            # denominators; they're accounted for separately in the Trimmer failure-codes
            # section.
            try:
                if rec.get_tag("RG") == "unmatched":
                    continue
            except KeyError:
                pass
            st = rec.get_tag(ST_TAG)  # KeyError if absent — st is required
            try:
                sr = float(rec.get_tag(SR_TAG))
            except KeyError:
                # sr is optional: libraries that don't carry it (e.g. legacy_v5 chemistries
                # without strand-ratio calibration) still produce a useful QC report, just
                # without the sr-based sections.
                sr = for_sr_nan
            try:
                et = rec.get_tag(ET_TAG)
            except KeyError:
                et = ""
            try:
                tm = rec.get_tag(TM_TAG)
            except KeyError:
                # A missing tm tag means the adapter was not reached (== END_UNREACHED
                # downstream), not "undetermined". See Ken's comment on PR #307.
                tm = ""
            read_len = rec.query_length or 0
            rows.append((sr, str(st), str(et), str(tm), int(read_len)))

    df_reads = pd.DataFrame(rows, columns=[SR_TAG, ST_TAG, ET_TAG, TM_TAG, "read_length"])
    df_reads[HistogramColumnNames.COUNT.value] = 1
    is_end_reached = df_reads[TM_TAG].str.contains("A", na=False)
    undetermined = PpmseqCategories.UNDETERMINED.value
    end_unreached = PpmseqCategories.END_UNREACHED.value
    df_reads[HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value] = df_reads[ST_TAG].replace("", undetermined)
    df_reads[HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value] = (
        df_reads[ET_TAG].where(is_end_reached, end_unreached).replace("", undetermined)
    )
    # Expose sr under STRAND_RATIO_{START,END} columns for downstream plotters that group
    # by those. Post-RAMP reads have a single sr per read; STRAND_RATIO_END is masked on
    # reads where the end was not reached.
    df_reads[HistogramColumnNames.STRAND_RATIO_START.value] = df_reads[SR_TAG].round(2)
    df_reads[HistogramColumnNames.STRAND_RATIO_END.value] = df_reads[SR_TAG].where(is_end_reached).round(2)
    df_reads[HistogramColumnNames.COUNT_NORM.value] = 1.0 / max(len(df_reads), 1)
    if sample_name:
        df_reads.index.name = sample_name
    if output_filename is not None:
        df_reads.to_parquet(output_filename)
    return df_reads


def has_sr_tag(df_reads: pd.DataFrame) -> bool:
    """Return True if at least one read in the per-read dataframe carries a non-NaN sr value."""
    return df_reads[SR_TAG].notna().any()


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
        per-read dataframe from read_tags_from_subsampled_sam (columns: sr, st, et, tm, count, ...)

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
        per-read dataframe from read_tags_from_subsampled_sam (columns: sr, st, et, tm, count, ...)

    Returns
    -------
    df_category_concordance: pd.DataFrame
        dataframe with strand ratio category columns as index and strand ratio category columns as columns
    df_category_concordance_no_end_unreached: pd.DataFrame
        dataframe with strand ratio category columns as index and strand ratio category columns as columns, excluding
        reads where the end was unreached


    """
    _assert_adapter_version_supported(adapter_version)
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

    """
    _assert_adapter_version_supported(adapter_version)
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
            # Keep the previous name for the consensus Series but expose it to the
            # shortlist as PCT_UNDETERMINED_end_tag (reviewer asked for the more
            # accurate name).
            undetermined: "PCT_UNDETERMINED_end_tag",
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
    mixed_start = df_category_concordance.loc[(PpmseqCategories.MIXED.value, slice(None)),].sum()
    df_mixed_cov = pd.DataFrame(
        {
            "PCT_MIXED_start_tag": mixed_start * 100,
            "PCT_MIXED_both_tags": mixed_tot * 100,
        },
        index=["value"],
    ).T["value"]
    df_tags = pd.concat((df_mixed_cov, df_tags))

    # Drop noise metrics from the headline table at the top of the report — kenissur
    # asked for these two to be removed from the shortlist. They stay available via the
    # /strand_ratio_category_consensus HDF5 key for downstream tooling / Papyrus.
    for drop_metric in ("PCT_UNDETERMINED_end_tag", "PCT_DISCORDANT"):
        if drop_metric in df_tags.index:
            df_tags = df_tags.drop(drop_metric)

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


def collect_statistics(
    adapter_version: str | PpmseqAdapterVersions,
    subsampled_sam: str,
    output_filename: str,
    sorter_stats_csv: str = None,
    trimmer_failure_codes_csv: str = None,
):
    """
    Collect statistics from a ppmSeq subsampled SAM file produced by sorter.

    Parameters
    ----------
    adapter_version : str | PpmseqAdapterVersions
        adapter version (kept for backward compatibility; SAM-based path treats all versions identically).
    subsampled_sam : str
        Path to the subsampled sam.gz produced by sorter when demux is called with ``--sample-nr-reads=N``.
    output_filename : str
        path to save dataframe to in hdf format (should end with .h5).
    sorter_stats_csv : str, optional
        path to a Sorter stats file; when present, its metrics are merged into the stats shortlist.
    trimmer_failure_codes_csv : str, optional
        path to a Trimmer failure codes file; when present, failure-rate metrics are appended.
    """
    _assert_adapter_version_supported(adapter_version)
    df_reads = read_tags_from_subsampled_sam(subsampled_sam)
    df_strand_ratio_category = group_trimmer_histogram_by_strand_ratio_category(adapter_version, df_reads)
    df_category_concordance, _, df_category_consensus = get_strand_ratio_category_concordance(adapter_version, df_reads)

    if sorter_stats_csv:
        # Read the raw two-column CSV directly instead of going through
        # read_and_parse_sorter_statistics_csv, which silently filters to a fixed set of
        # ~7 metrics and would drop several of the rows we want to show
        # (e.g. PCT_SOFTCLIPPED_bases, PCT_Adapter_Reached).
        raw = pd.read_csv(sorter_stats_csv, header=None, names=["metric", "value"])
        raw = raw.dropna(subset=["metric"]).set_index("metric")["value"]
        # Coerce to float; any value that fails (e.g. F80@30x's "(1.43)" strings) becomes NaN
        # and is then dropped — we only care about the numeric allow-listed keys.
        raw = pd.to_numeric(raw, errors="coerce").dropna()
        # Keep only the allow-listed keys, silently drop anything else. Preserve the
        # canonical order so the report table reads top-to-bottom in the expected layout.
        present = [k for k in SORTER_STATS_KEYS_TO_SHOW if k in raw.index]
        sorter_stats = raw.reindex(present)
        sorter_stats.name = "value"
        sorter_stats.index.name = "metric"
    else:
        sorter_stats = pd.Series(dtype=float, name="value")
        sorter_stats.index.name = "metric"

    df_tags = read_trimmer_tags_dataframe(
        adapter_version=adapter_version,
        df_strand_ratio_category=df_strand_ratio_category,
        df_category_consensus=df_category_consensus,
        df_sorter_stats=sorter_stats.to_frame(),
        df_category_concordance=df_category_concordance,
    )

    # stats_shortlist holds the ppmSeq-specific metrics (MIXED coverage / strand-ratio
    # category percentages). Sorter stats are stored under a separate /sorter_stats key
    # so the report can show them in a dedicated table and users asking for "mixed reads
    # stats" aren't buried under the sorter aggregates.
    df_stats_shortlist = df_tags

    if trimmer_failure_codes_csv:
        df_trimmer_failure_codes, df_failure_codes_metrics = read_trimmer_failure_codes_ppmseq(
            trimmer_failure_codes_csv
        )

    if not output_filename.endswith(".h5"):
        output_filename += ".h5"
    with pd.HDFStore(output_filename, "w") as store:
        keys_to_convert = [
            "stats_shortlist",
            "sorter_stats",
            "strand_ratio_category_counts",
            "strand_ratio_category_norm",
            "strand_ratio_category_concordance",
            "strand_ratio_category_consensus",
        ]
        store["stats_shortlist"] = df_stats_shortlist
        store["sorter_stats"] = sorter_stats
        store["subsampled_reads"] = df_reads
        store["strand_ratio_category_counts"] = df_strand_ratio_category
        store["strand_ratio_category_norm"] = df_strand_ratio_category / df_strand_ratio_category.sum()
        store["strand_ratio_category_concordance"] = df_category_concordance
        store["strand_ratio_category_consensus"] = df_category_consensus
        if trimmer_failure_codes_csv:
            store["trimmer_failure_codes"] = df_trimmer_failure_codes
            store["failure_codes_metrics"] = df_failure_codes_metrics
            keys_to_convert += ["trimmer_failure_codes", "failure_codes_metrics"]
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


def plot_strand_ratio_category(
    adapter_version: str | PpmseqAdapterVersions,
    df_trimmer_histogram: pd.DataFrame,
    title: str = "",
    output_filename: str = None,
    ax: plt.Axes = None,
    *,
    sr_present: bool = False,
) -> plt.Axes:
    """
    Plot the strand ratio category histogram

    Parameters
    ----------
    adapter_version : str | PpmseqAdapterVersions
        adapter version to check
    df_trimmer_histogram : pd.DataFrame
        per-read dataframe from read_tags_from_subsampled_sam (columns: sr, st, et, tm, count, ...)
    title : str, optional
        plot title, by default ""
    output_filename : str, optional
        path to save the plot to, by default None (not saved)
    ax : matplotlib.axes.Axes, optional
        axes to plot on, by default None (new figure created)
    sr_present : bool, optional
        When True, drop the UNDETERMINED bar from the Start-tag axis — when sr is
        available on every read the Trimmer --compare cascade only emits
        MIXED / PLUS / MINUS, so UNDETERMINED on the start axis is by construction
        empty and showing it is misleading. The end-tag axis still keeps UNDETERMINED
        because et comes from base calling regardless of sr.

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
    normalized = df_trimmer_histogram_by_strand_ratio_category / df_trimmer_histogram_by_strand_ratio_category.sum()
    if sr_present:
        # Zero out UNDETERMINED on the Start-tag column only (if present); leave the End-tag
        # column untouched because et can still be UNDETERMINED when sr is present.
        start_col = "Start tag strand ratio category"
        undetermined = PpmseqCategories.UNDETERMINED.value
        if start_col in normalized.columns and undetermined in normalized.index:
            normalized.loc[undetermined, start_col] = float("nan")
    df_plot = normalized.reset_index().melt(id_vars="index", var_name="").dropna(subset=["value"])
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
    *,
    sr_present: bool = False,
) -> list[plt.Axes]:
    """
    Plot the strand ratio category concordance heatmap

    Parameters
    ----------
    adapter_version : str | PpmseqAdapterVersions
        adapter version to check
    df_trimmer_histogram : pd.DataFrame
        per-read dataframe from read_tags_from_subsampled_sam (columns: sr, st, et, tm, count, ...)
    title : str, optional
        plot title, by default ""
    output_filename : str, optional
        path to save the plot to, by default None (not saved)
    axs : matplotlib.axes.Axes, optional
        axes to plot on, by default None (new figure created)
    sr_present : bool, optional
        When True, drop the UNDETERMINED row from the start-tag axis — see
        plot_strand_ratio_category for rationale. The end-tag (column) axis still
        keeps UNDETERMINED.

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
    # Start-axis categories. UNDETERMINED is dropped when sr is present because the
    # sr-based cascade never emits UNDETERMINED on the start axis.
    start_categories = [v for v in ppmseq_category_list if v != PpmseqCategories.END_UNREACHED.value]
    if sr_present:
        start_categories = [v for v in start_categories if v != PpmseqCategories.UNDETERMINED.value]
    # plot
    for ax, subtitle, df_plot in zip(
        axs,
        ("All reads", "Only reads where end was reached"),
        (df_category_concordance, df_category_concordance_no_end_unreached),
        strict=False,
    ):
        df_plot = df_plot.to_frame().unstack().droplevel(0, axis=1)  # noqa: PD010, PLW2901
        df_plot = df_plot.loc[  # noqa: PLW2901
            [v for v in start_categories if v in df_plot.index],
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


_SR_BIN_EDGES = np.arange(0.0, 1.01, 0.01)
_SR_TITLE_FONTSIZE = 28
_SR_AXIS_LABEL_FONTSIZE = 20

# Vertical lines at the sr thresholds used by the Solaris-2 --compare cascade (MIXED band edges).
_SR_THRESHOLD_LINES = (0.2, 0.8)

# Colors for sr-by-et facets; one per end-tag category so the per-category panels read at a glance.
_SR_CATEGORY_COLORS = {
    PpmseqCategories.PLUS.value: "xkcd:royal blue",
    PpmseqCategories.MINUS.value: "xkcd:red orange",
    PpmseqCategories.MIXED.value: "xkcd:green",
    PpmseqCategories.UNDETERMINED.value: "xkcd:grey",
    PpmseqCategories.END_UNREACHED.value: "xkcd:black",
}


def _plot_sr_hist_on_ax(ax: plt.Axes, sr_values: pd.Series, color: str) -> None:
    """Draw an sr histogram on the given ax as a line plot over bin centers, expressed as a
    percentage so facets are directly comparable. Values outside [0, 1] are clipped
    (np.clip-style) so the outlier mass lands in the edge bins rather than being discarded."""
    n = len(sr_values)
    if n == 0:
        ax.text(0.5, 0.5, "no reads", transform=ax.transAxes, ha="center", va="center")
        return
    clipped = sr_values.clip(lower=_SR_BIN_EDGES[0], upper=_SR_BIN_EDGES[-1])
    counts, _ = np.histogram(clipped, bins=_SR_BIN_EDGES)
    percent_per_bin = 100.0 * counts / n
    centers = 0.5 * (_SR_BIN_EDGES[:-1] + _SR_BIN_EDGES[1:])
    ax.plot(centers, percent_per_bin, color=color, linewidth=2)
    ax.set_xlim(_SR_BIN_EDGES[0], _SR_BIN_EDGES[-1])
    for x in _SR_THRESHOLD_LINES:
        ax.axvline(x, color="xkcd:dark grey", linestyle="--", linewidth=1.2)


def plot_sr_histogram(
    df_reads: pd.DataFrame,
    title: str = "",
    output_filename: str = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Plot the overall strand-ratio histogram across all reads. Bins are 0.01 wide and the
    y-axis is a percentage of total reads. Reads with sr outside [0, 1] are clipped to the
    edge bins so they remain visible. Two dashed guide lines at 0.2 and 0.8 mark the
    thresholds used by the Solaris-2 ``--compare`` cascade.

    Parameters
    ----------
    df_reads : pd.DataFrame
        Per-read tag dataframe from :func:`read_tags_from_subsampled_sam`.
    title : str, optional
        plot title.
    output_filename : str, optional
        if provided, save the plot.
    ax : matplotlib.axes.Axes, optional
        axes to plot on.
    """
    set_pyplot_defaults()
    if ax is None:
        plt.figure(figsize=(12, 4))
        ax = plt.gca()
    else:
        plt.sca(ax)
    sr_values = df_reads[SR_TAG].dropna()
    _plot_sr_hist_on_ax(ax, sr_values, "xkcd:royal blue")
    ax.set_xlabel(STRAND_RATIO_AXIS_LABEL, fontsize=_SR_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Frequency (%)", fontsize=_SR_AXIS_LABEL_FONTSIZE)
    title_handle = plt.title(title, fontsize=_SR_TITLE_FONTSIZE)
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
    return ax


def plot_sr_by_et(
    df_reads: pd.DataFrame,
    title: str = "",
    output_filename: str = None,
) -> plt.Figure:
    """
    Plot strand-ratio histograms faceted by the end-loop category, as a vertically stacked
    4x1 grid (one panel per category: PLUS / MINUS / MIXED / UNDETERMINED). Values are a
    percentage of reads within each panel. Restricted to reads whose read end was reached.
    Reads with sr outside [0, 1] are clipped (np.clip-style) so they remain visible.

    Parameters
    ----------
    df_reads : pd.DataFrame
        Per-read tag dataframe from :func:`read_tags_from_subsampled_sam`.
    title : str, optional
        plot title.
    output_filename : str, optional
        if provided, save the plot.
    """
    set_pyplot_defaults()
    # Use the filled strand_ratio_category_end column so reads whose end was reached but
    # whose et tag was missing show up under UNDETERMINED instead of being silently dropped.
    # UNDETERMINED reads are excluded from the panels (the reviewer asked for it, and in
    # practice end-reached reads with UNDETERMINED end-tag are rare).
    category_col = HistogramColumnNames.STRAND_RATIO_CATEGORY_END.value
    df_endreached = df_reads[df_reads[TM_TAG].str.contains("A", na=False)]
    et_categories = [
        PpmseqCategories.MIXED.value,
        PpmseqCategories.PLUS.value,
        PpmseqCategories.MINUS.value,
    ]
    # Vertical stack: each panel the same size as the overall plot (width 12, height 4).
    fig, axs = plt.subplots(len(et_categories), 1, figsize=(12, 4 * len(et_categories)), sharex=True)
    for ax, et_val in zip(axs, et_categories, strict=True):
        subset = df_endreached[df_endreached[category_col] == et_val]
        sr_values = subset[SR_TAG].dropna()
        _plot_sr_hist_on_ax(ax, sr_values, _SR_CATEGORY_COLORS[et_val])
        ax.set_title(f"et={et_val}", fontsize=_SR_TITLE_FONTSIZE)
        ax.set_ylabel("Frequency (%)", fontsize=_SR_AXIS_LABEL_FONTSIZE)
    axs[-1].set_xlabel(STRAND_RATIO_AXIS_LABEL, fontsize=_SR_AXIS_LABEL_FONTSIZE)
    title_handle = fig.suptitle(title, fontsize=_SR_TITLE_FONTSIZE)
    fig.tight_layout()
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
    return fig


def plot_read_length_by_st(
    df_reads: pd.DataFrame,
    title: str = "",
    output_filename: str = None,
) -> plt.Figure:
    """
    Plot per-read read-length distributions faceted by the start-loop category as a vertically
    stacked 3x1 grid (PLUS / MINUS / MIXED). Values are a percentage of reads within each panel.
    One color per category (shared with plot_sr_by_et). UNDETERMINED reads are excluded because
    in practice almost no reads carry that category at this point in the pipeline.

    Parameters
    ----------
    df_reads : pd.DataFrame
        Per-read tag dataframe from :func:`read_tags_from_subsampled_sam`; must have a
        ``read_length`` column.
    title : str, optional
        plot title.
    output_filename : str, optional
        if provided, save the plot.
    """
    set_pyplot_defaults()
    category_col = HistogramColumnNames.STRAND_RATIO_CATEGORY_START.value
    st_categories = [
        PpmseqCategories.MIXED.value,
        PpmseqCategories.PLUS.value,
        PpmseqCategories.MINUS.value,
    ]
    # Shared x range so facets are directly comparable. 99th percentile trims the long tail.
    lengths_all = df_reads["read_length"].dropna()
    x_max = int(np.ceil(lengths_all.quantile(0.995))) if len(lengths_all) else 1
    x_max = max(x_max, 10)
    # 1 bp bins centered on integer read lengths — with edges at .5 offsets each bin
    # covers exactly one integer bp value, so the centers land on N (not N + 0.5) and
    # every bin holds exactly one read-length value (no zig-zag).
    bin_edges = np.arange(-0.5, x_max + 0.5, 1.0)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, axs = plt.subplots(len(st_categories), 1, figsize=(12, 4 * len(st_categories)), sharex=True)
    for ax, st_val in zip(axs, st_categories, strict=True):
        subset = df_reads[df_reads[category_col] == st_val]
        lengths = subset["read_length"].dropna()
        n = len(lengths)
        if n == 0:
            ax.text(0.5, 0.5, "no reads", transform=ax.transAxes, ha="center", va="center")
        else:
            clipped = lengths.clip(lower=0, upper=x_max)
            counts, _ = np.histogram(clipped, bins=bin_edges)
            percent_per_bin = 100.0 * counts / n
            ax.plot(centers, percent_per_bin, color=_SR_CATEGORY_COLORS[st_val], linewidth=2)
        ax.set_xlim(0, x_max)
        ax.set_title(f"st={st_val}", fontsize=_SR_TITLE_FONTSIZE)
        ax.set_ylabel("Frequency (%)", fontsize=_SR_AXIS_LABEL_FONTSIZE)
    axs[-1].set_xlabel("Read length (bp)", fontsize=_SR_AXIS_LABEL_FONTSIZE)
    title_handle = fig.suptitle(title, fontsize=_SR_TITLE_FONTSIZE)
    fig.tight_layout()
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
    return fig


def plot_read_length_overall(
    df_reads: pd.DataFrame,
    title: str = "",
    output_filename: str = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Plot the overall read-length histogram as a line plot over bin centers.

    Parameters
    ----------
    df_reads : pd.DataFrame
        Per-read tag dataframe from :func:`read_tags_from_subsampled_sam`; must have a
        ``read_length`` column.
    title : str, optional
        plot title.
    output_filename : str, optional
        if provided, save the plot.
    ax : matplotlib.axes.Axes, optional
        axes to plot on.
    """
    set_pyplot_defaults()
    if ax is None:
        plt.figure(figsize=(12, 4))
        ax = plt.gca()
    else:
        plt.sca(ax)
    lengths = df_reads["read_length"].dropna()
    n = len(lengths)
    x_max = int(np.ceil(lengths.quantile(0.995))) if n else 1
    x_max = max(x_max, 10)
    bin_edges = np.arange(-0.5, x_max + 0.5, 1.0)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if n:
        clipped = lengths.clip(lower=0, upper=x_max)
        counts, _ = np.histogram(clipped, bins=bin_edges)
        ax.plot(centers, 100.0 * counts / n, color="xkcd:royal blue", linewidth=2)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Read length (bp)", fontsize=_SR_AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Frequency (%)", fontsize=_SR_AXIS_LABEL_FONTSIZE)
    title_handle = plt.title(title, fontsize=_SR_TITLE_FONTSIZE)
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
    return ax


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


def ppmseq_qc_analysis(  # noqa: PLR0913
    adapter_version: str | PpmseqAdapterVersions,
    subsampled_sam: str,
    output_path: str,
    output_basename: str = None,
    sorter_stats_csv: str = None,
    sorter_stats_json: str = None,
    trimmer_failure_codes_csv: str = None,
    generate_report: bool = True,  # noqa: FBT001, FBT002
    keep_temp_visualization_files: bool = False,  # noqa: FBT001, FBT002
    qc_filename_suffix: str = ".ppmSeq.applicationQC",
    extra_info: dict = None,
):
    """
    Run the ppmSeq QC analysis pipeline on the subsampled SAM produced by sorter.

    Parameters
    ----------
    adapter_version : str | PpmseqAdapterVersions
        adapter version.
    subsampled_sam : str
        Path to the sorter-produced subsampled sam, bam, or cram file (requires demux
        --sample-nr-reads=N). Pysam picks the format from the extension.
    output_path : str
        output folder for h5, json, and html.
    output_basename : str, optional
        basename for output files; defaults to the SAM file basename.
    sorter_stats_csv : str, optional
        path to a Sorter stats csv file.
    sorter_stats_json : str, optional
        accepted for backwards compat; currently unused (the sorter-reported read length
        plot was removed because the per-read read-length plots in Section 3 are
        authoritative for this report).
    trimmer_failure_codes_csv : str, optional
        path to a Trimmer failure codes csv file. If provided, failure-rate metrics are added.
    generate_report : bool, optional
        if True, generate an html + jupyter report.
    keep_temp_visualization_files : bool, optional
        if True, keep temporary png/ipynb files.
    qc_filename_suffix : str, optional
        suffix for the output statistics file immediately after output_basename.
    extra_info : dict, optional
        Free-form key/value pairs to surface at the top of the HTML report (e.g.
        ``{"version": "1.2.3.4"}``). Rendered as a small table between the sample
        header and the table of contents.
    """
    del sorter_stats_json  # retained in signature for backwards compat; no longer rendered
    _assert_adapter_version_supported(adapter_version)
    if not os.path.isfile(subsampled_sam):
        raise FileNotFoundError(f"{subsampled_sam} not found")

    os.makedirs(output_path, exist_ok=True)
    if output_basename is None:
        output_basename = os.path.basename(subsampled_sam)

    output_statistics_h5 = os.path.join(output_path, f"{output_basename}{qc_filename_suffix}.h5")
    output_statistics_json = os.path.join(output_path, f"{output_basename}{qc_filename_suffix}.json")
    output_report_html = Path(output_path) / f"{output_basename}{qc_filename_suffix}.html"
    output_report_ipynb = os.path.join(output_path, f"{output_basename}{qc_filename_suffix}.ipynb")
    output_strand_ratio_category_plot = os.path.join(output_path, f"{output_basename}.strand_ratio_category.png")
    output_strand_ratio_category_concordance_plot = os.path.join(
        output_path, f"{output_basename}.strand_ratio_category_concordance.png"
    )
    output_sr_hist_plot = os.path.join(output_path, f"{output_basename}.sr_hist.png")
    output_sr_by_et_plot = os.path.join(output_path, f"{output_basename}.sr_by_et.png")
    output_read_length_plot = os.path.join(output_path, f"{output_basename}.read_length.png")
    output_read_length_by_st_plot = os.path.join(output_path, f"{output_basename}.read_length_by_st.png")
    output_visualization_files = [
        output_report_ipynb,
        output_strand_ratio_category_plot,
        output_strand_ratio_category_concordance_plot,
        output_read_length_plot,
        output_read_length_by_st_plot,
    ]

    collect_statistics(
        adapter_version=adapter_version,
        subsampled_sam=subsampled_sam,
        sorter_stats_csv=sorter_stats_csv,
        trimmer_failure_codes_csv=trimmer_failure_codes_csv,
        output_filename=output_statistics_h5,
    )

    df_reads = pd.read_hdf(output_statistics_h5, "subsampled_reads")
    convert_h5_to_papyrus_json(output_statistics_h5, output_statistics_json)
    sr_present = has_sr_tag(df_reads)

    plot_strand_ratio_category(
        adapter_version,
        df_reads,
        title=f"{output_basename} strand ratio category",
        output_filename=output_strand_ratio_category_plot,
        sr_present=sr_present,
    )
    plot_strand_ratio_category_concordnace(
        adapter_version,
        df_reads,
        title=f"{output_basename} strand ratio category concordance",
        output_filename=output_strand_ratio_category_concordance_plot,
        sr_present=sr_present,
    )
    if sr_present:
        plot_sr_histogram(
            df_reads,
            title=f"{output_basename} strand ratio",
            output_filename=output_sr_hist_plot,
        )
        plot_sr_by_et(
            df_reads,
            title=f"{output_basename} strand ratio by et (end reached)",
            output_filename=output_sr_by_et_plot,
        )
        output_visualization_files.extend([output_sr_hist_plot, output_sr_by_et_plot])
    plot_read_length_overall(
        df_reads,
        title=f"{output_basename} read length",
        output_filename=output_read_length_plot,
    )
    plot_read_length_by_st(
        df_reads,
        title=f"{output_basename} read length by st",
        output_filename=output_read_length_by_st_plot,
    )

    if generate_report:
        template_notebook = BASE_PATH / REPORTS_DIR / "ppmSeq_qc_report.ipynb"
        adapter_value = adapter_version if isinstance(adapter_version, str) else adapter_version.value
        logo_path = BASE_PATH / REPORTS_DIR / "ug_logo.b64"
        parameters = {
            "sample_name": output_basename,
            # The notebook still validates that adapter_version is supported so we keep
            # passing it through, but the report itself never displays it.
            "adapter_version": adapter_value,
            "statistics_h5": output_statistics_h5,
            "strand_ratio_category_png": output_strand_ratio_category_plot,
            "strand_ratio_category_concordance_png": output_strand_ratio_category_concordance_plot,
            # When sr is absent the sr-dependent sections (1.4 and 1.5) are dropped from the
            # report. Passing None here signals the notebook to skip them.
            "sr_hist_png": output_sr_hist_plot if sr_present else None,
            "sr_by_et_png": output_sr_by_et_plot if sr_present else None,
            "read_length_png": output_read_length_plot,
            "read_length_by_st_png": output_read_length_by_st_plot,
            "logo_file": str(logo_path),
            "trimmer_failure_codes_csv": trimmer_failure_codes_csv,
            "sorter_stats_csv": sorter_stats_csv,
            "extra_info": dict(extra_info) if extra_info else {},
        }

        if not keep_temp_visualization_files:
            tmp_files = [Path(file) for file in output_visualization_files]
        else:
            tmp_files = None

        generate_report_func(
            template_notebook_path=template_notebook,
            parameters=parameters,
            output_report_html_path=output_report_html,
            tmp_files=tmp_files,
        )
