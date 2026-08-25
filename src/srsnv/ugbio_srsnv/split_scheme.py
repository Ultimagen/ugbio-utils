"""Data-driven read-split "recipes" for the SRSNV report.

The report splits reads into groups (e.g. ppmSeq mixed/non-mixed, or consensus
single/one-strand/duplex) for every figure and table. Instead of scattering ``if mode ==``
branches across the report methods, all split behavior is described by a :class:`SplitScheme`
"recipe": a detector predicate, a function that adds the columns the scheme needs, and an
ordered list of :class:`SplitVariant` renderings. Each figure/table method consumes the active
scheme's variants uniformly and never names a scheme.

Adding a new split scheme = define one :class:`SplitScheme` object (+ its detector) and append it
to :data:`SPLIT_SCHEMES`. No report method changes.

h5 keys: every split table is written as ``base`` for the legacy variant (``suffix == ""``) and
``base + "_" + suffix`` for other variants (see :func:`h5_key`). Mixed keeps its historical
suffixes (``""`` legacy / ``mixed_start`` display); other schemes get their own scheme-id suffix.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from ugbio_core.logger import logger

from ugbio_srsnv.srsnv_utils import (
    AE,
    AS,
    CONSENSUS_GROUP_DUPLEX,
    CONSENSUS_GROUP_ONE_STRAND,
    CONSENSUS_GROUP_SINGLE,
    CONSENSUS_GROUPS,
    DUPLEX_MOL_GROUPS,
    DUPLEX_MOL_PAIRED,
    DUPLEX_MOL_SINGLE,
    ET,
    FS,
    IS_CONSENSUS,
    IS_MIXED,
    IS_MIXED_START,
    MATE_PRESENT,
    MIXED_GROUP_NON,
    MIXED_GROUP_POS,
    MIXED_GROUPS,
    NONE_GROUP_ALL,
    READ_GROUP,
    RS,
    ST,
    TE,
    TS,
    ReportMode,
    add_duplex_columns_to_featuremap_df,
    add_is_consensus_to_featuremap_df,
    add_is_mixed_to_featuremap_df,
)

MIN_CONSENSUS_READS = 2  # fs + rs >= 2 to be more than a single read


@dataclass(frozen=True)
class SplitVariant:
    """One rendering of the read split into ordered groups, with its own h5-key suffix.

    A scheme has one or more variants. The report loops variants: each writes its table under
    ``h5_key(base, variant)`` and the ``is_display`` variant additionally drives the PNG and the
    key the notebook reads.

    Attributes
    ----------
    suffix : str
        h5-key suffix. ``""`` yields the bare ``base`` key (the historical "legacy" key).
    group_fn : Callable[[pd.DataFrame], pd.Series]
        Maps rows to ordered group labels (returns an ordered Categorical / label Series aligned
        to ``df.index``). The single source of truth for how this variant groups reads.
    groups : tuple[str, ...]
        Ordered group labels this variant emits (2 for a binary split, N for N groups).
    colors : dict | None
        ``{label: matplotlib color}``; ``None`` -> seaborn palette sized to ``groups``.
    is_display : bool
        Exactly one variant per scheme is the display variant (drives PNG + notebook key).
    pos, neg, pos_lc, neg_lc : str | None
        Binary-framing labels, used only by renderers that still need a positive/negative framing
        (ROC "X only" rows, run-quality columns) for byte-identical mixed output. ``None`` for
        purely N-group schemes -> those renderers use the N-group path (``pos is None`` discriminator).
    tag_axis : str
        Axis/index label for the per-tag summary table.
    hist_col_name : str
        ``col_name`` passed to the histogram-data extractor; part of the h5 column MultiIndex, so it
        must stay literal for byte-identical mixed output. Empty -> use the group column name.
    fq_conditions_fn : Callable | None
        Optional builder of the ordered ``{label: mask|None}`` FQ-recall conditions for this variant
        (mixed legacy needs the special both-ends/start conditions). ``None`` -> derived from groups.
    summary_group_fn : Callable | None
        Optional grouping for the per-tag summary table when it differs from ``group_fn`` (mixed
        display groups by start-tag category, not by the 2-group label). ``None`` -> ``group_fn``.
    trinuc_split_col : str | None
        Column name fed to the trinuc plotter as its split column. ``None`` -> a materialized
        ``read_group`` column. (Mixed uses the raw ``is_mixed`` bool to keep trinuc_stats identical.)
    trinuc_group_specs : Any
        ``group_specs`` passed to the trinuc plotter (``None`` -> the plotter auto-detects, which is
        the historical mixed behavior).
    """

    suffix: str
    group_fn: Callable[[pd.DataFrame], pd.Series]
    groups: tuple[str, ...]
    colors: dict[str, Any] | None = None
    is_display: bool = False
    pos: str | None = None
    neg: str | None = None
    pos_lc: str | None = None
    neg_lc: str | None = None
    tag_axis: str = ""
    hist_col_name: str = ""
    fq_conditions_fn: Callable[[pd.DataFrame], dict] | None = None
    summary_group_fn: Callable[[pd.DataFrame], pd.Series] | None = None
    trinuc_split_col: str | None = None
    trinuc_group_specs: Any = None

    def group_masks(self, data_df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Ordered ``{label: boolean mask (np.ndarray)}`` for this variant's groups over ``data_df``."""
        col = np.asarray(self.group_fn(data_df))
        return {label: (col == label) for label in self.groups}


@dataclass(frozen=True)
class SplitScheme:
    """A full read-split recipe for one detected data type.

    Attributes
    ----------
    mode : ReportMode
        Detection key / legacy mode id (its ``.value`` is stored in metadata JSON). Report methods
        never switch on this.
    detect : Callable[[pd.Index], bool]
        Column-presence predicate deciding whether this scheme applies to a dataframe.
    add_columns : Callable[[pd.DataFrame, dict], pd.DataFrame]
        Adds the boolean / categorical columns the variants' ``group_fn``s read.
    variants : tuple[SplitVariant, ...]
        Ordered; ``variants[0]`` is the legacy/base-key variant, one variant has ``is_display``.
    tag_axis : str
        Per-tag summary axis label ("ppmSeq tag" / "strand support").
    legacy_hist_fn, crosstab_fn : Callable | None
        Optional scheme-owned renderers for the two genuinely ppmSeq-specific outputs (the
        3-category quality histogram and the start x end 2D tables). ``None`` -> generic path.
    """

    mode: ReportMode
    detect: Callable[[pd.Index], bool]
    add_columns: Callable[[pd.DataFrame, dict], pd.DataFrame]
    variants: tuple[SplitVariant, ...]
    tag_axis: str = ""
    # Capability flags for the two genuinely ppmSeq-specific renderers (True only for mixed).
    # The report dispatches on these instead of checking the mode.
    ppmseq_crosstab: bool = False  # per-tag table uses the start x end 2D cross-tab
    ppmseq_legacy_hist: bool = False  # quality histogram uses the 3-category (non/one-end/both-ends) split

    @property
    def display_variant(self) -> SplitVariant:
        return next(v for v in self.variants if v.is_display)

    @property
    def legacy_variant(self) -> SplitVariant:
        return self.variants[0]

    @property
    def display_suffix(self) -> str:
        return self.display_variant.suffix


def h5_key(base: str, variant: SplitVariant) -> str:
    """Compose the h5 key for a base name and variant: bare base for empty suffix, else suffixed."""
    return f"{base}_{variant.suffix}" if variant.suffix else base


# ──────────────────────────── group functions ──────────────────────────────


def _ordered(labels: pd.Series | np.ndarray, categories: list[str]) -> pd.Categorical:
    return pd.Categorical(labels, categories=categories, ordered=True)


def _mixed_display_groups(data_df: pd.DataFrame) -> pd.Categorical:
    """Start-tag split -> [Non-mixed, Mixed] (the display split)."""
    g = pd.Series(MIXED_GROUP_NON, index=data_df.index, dtype=object)
    g[data_df[IS_MIXED_START]] = MIXED_GROUP_POS
    return _ordered(g, MIXED_GROUPS)


def _mixed_legacy_groups(data_df: pd.DataFrame) -> pd.Categorical:
    """Both-ends split -> [Non-mixed, Mixed] (the legacy split)."""
    g = pd.Series(MIXED_GROUP_NON, index=data_df.index, dtype=object)
    g[data_df[IS_MIXED]] = MIXED_GROUP_POS
    return _ordered(g, MIXED_GROUPS)


def _consensus_groups(data_df: pd.DataFrame) -> pd.Categorical:
    """Strand-support split: single read / consensus one strand / consensus duplex."""
    fs = data_df[FS]
    rs = data_df[RS]
    total = fs + rs
    g = pd.Series(CONSENSUS_GROUP_SINGLE, index=data_df.index, dtype=object)
    g[(total >= MIN_CONSENSUS_READS) & ((fs == 0) ^ (rs == 0))] = CONSENSUS_GROUP_ONE_STRAND
    g[(fs >= 1) & (rs >= 1)] = CONSENSUS_GROUP_DUPLEX
    return _ordered(g, CONSENSUS_GROUPS)


def _duplex_groups(data_df: pd.DataFrame) -> pd.Categorical:
    """Per-molecule split: single-strand molecule / duplex molecule (from ``mate_present``)."""
    g = pd.Series(DUPLEX_MOL_SINGLE, index=data_df.index, dtype=object)
    g[data_df[MATE_PRESENT].astype(bool)] = DUPLEX_MOL_PAIRED
    return _ordered(g, DUPLEX_MOL_GROUPS)


def _none_groups(data_df: pd.DataFrame) -> pd.Categorical:
    return _ordered([NONE_GROUP_ALL] * len(data_df), [NONE_GROUP_ALL])


# ──────────────────────────── add-columns hooks ────────────────────────────


def _mixed_add_columns(data_df: pd.DataFrame, kw: dict) -> pd.DataFrame:
    return add_is_mixed_to_featuremap_df(data_df, kw.get("adapter_version"), kw.get("categorical_features_names"))


def _consensus_add_columns(data_df: pd.DataFrame, kw: dict) -> pd.DataFrame:  # noqa: ARG001
    return add_is_consensus_to_featuremap_df(data_df)


def _duplex_add_columns(data_df: pd.DataFrame, kw: dict) -> pd.DataFrame:  # noqa: ARG001
    return add_duplex_columns_to_featuremap_df(data_df)


def _none_add_columns(data_df: pd.DataFrame, kw: dict) -> pd.DataFrame:  # noqa: ARG001
    data_df[IS_CONSENSUS] = False
    return data_df


# ──────────────────── mixed FQ-recall condition builders ────────────────────
# These reproduce the historical ppmSeq FQ-recall conditions exactly (the only place the raw
# is_mixed / is_mixed_start columns are read for FQ-recall).


def _mixed_legacy_fq_conditions(data_df: pd.DataFrame) -> dict:
    conditions = {"all reads": None}
    if IS_MIXED in data_df.columns and data_df[IS_MIXED].any():
        conditions["mixed both ends"] = data_df[IS_MIXED]
    if IS_MIXED_START in data_df.columns and data_df[IS_MIXED_START].any():
        conditions["mixed start"] = data_df[IS_MIXED_START]
    return conditions


def _mixed_display_fq_conditions(data_df: pd.DataFrame) -> dict:
    conditions = {"all reads": None}
    if IS_MIXED_START in data_df.columns and data_df[IS_MIXED_START].any():
        conditions["mixed"] = data_df[IS_MIXED_START]
    return conditions


# ──────────────────────────── the recipes ──────────────────────────────────

MIXED_SCHEME = SplitScheme(
    mode=ReportMode.MIXED,
    detect=lambda cols: (ST in cols and ET in cols) or all(c in cols for c in (AS, AE, TS, TE)),
    add_columns=_mixed_add_columns,
    tag_axis="ppmSeq tag",
    ppmseq_crosstab=True,
    ppmseq_legacy_hist=True,
    variants=(
        SplitVariant(
            suffix="",
            group_fn=_mixed_legacy_groups,
            groups=tuple(MIXED_GROUPS),
            colors={MIXED_GROUP_NON: "tab:red", MIXED_GROUP_POS: "tab:green"},
            pos=MIXED_GROUP_POS,
            neg=MIXED_GROUP_NON,
            pos_lc="mixed",
            neg_lc="non-mixed",
            tag_axis="ppmSeq tag",
            hist_col_name=IS_MIXED,
            trinuc_split_col=IS_MIXED,
            trinuc_group_specs=None,
            fq_conditions_fn=_mixed_legacy_fq_conditions,
        ),
        SplitVariant(
            suffix="mixed_start",
            group_fn=_mixed_display_groups,
            groups=tuple(MIXED_GROUPS),
            colors={MIXED_GROUP_NON: "tab:red", MIXED_GROUP_POS: "tab:green"},
            is_display=True,
            pos=MIXED_GROUP_POS,
            neg=MIXED_GROUP_NON,
            pos_lc="mixed",
            neg_lc="non-mixed",
            tag_axis="ppmSeq tag",
            hist_col_name=IS_MIXED_START,
            trinuc_split_col=IS_MIXED,
            trinuc_group_specs=None,
            fq_conditions_fn=_mixed_display_fq_conditions,
        ),
    ),
)

DUPLEX_SCHEME = SplitScheme(
    mode=ReportMode.DUPLEX,
    detect=lambda cols: MATE_PRESENT in cols,
    add_columns=_duplex_add_columns,
    tag_axis="mate status",
    variants=(
        SplitVariant(
            suffix="duplex_molecule",
            group_fn=_duplex_groups,
            groups=tuple(DUPLEX_MOL_GROUPS),
            is_display=True,
            tag_axis="mate status",
        ),
    ),
)

CONSENSUS_SCHEME = SplitScheme(
    mode=ReportMode.CONSENSUS,
    detect=lambda cols: FS in cols and RS in cols,
    add_columns=_consensus_add_columns,
    tag_axis="strand support",
    variants=(
        SplitVariant(
            suffix="strand_support",
            group_fn=_consensus_groups,
            groups=tuple(CONSENSUS_GROUPS),
            is_display=True,
            tag_axis="strand support",
        ),
    ),
)

NONE_SCHEME = SplitScheme(
    mode=ReportMode.NONE,
    detect=lambda cols: True,  # fallback; never reached via detection order
    add_columns=_none_add_columns,
    tag_axis="ppmSeq tag",
    variants=(
        SplitVariant(
            suffix="",
            group_fn=_none_groups,
            groups=(NONE_GROUP_ALL,),
            is_display=True,
            tag_axis="ppmSeq tag",
        ),
    ),
)

# Ordered registry. Detection order preserves the historical priority: ppmSeq tags beat everything;
# the per-molecule duplex `mate_present` flag beats per-read fs/rs (a duplex featuremap may carry
# both, and the per-molecule split must win). NONE is the explicit fallback, not consulted via
# `detect`.
SPLIT_SCHEMES: tuple[SplitScheme, ...] = (MIXED_SCHEME, DUPLEX_SCHEME, CONSENSUS_SCHEME, NONE_SCHEME)
_BY_MODE: dict[ReportMode, SplitScheme] = {s.mode: s for s in SPLIT_SCHEMES}


def resolve_scheme(data_df: pd.DataFrame | None = None, *, mode: ReportMode | str | None = None) -> SplitScheme:
    """Return the active split scheme.

    If ``mode`` is given (a :class:`ReportMode` or its string value), return that scheme directly
    (used when the mode was already resolved and stored in params). Otherwise detect from the
    dataframe columns, in registry order, falling back to :data:`NONE_SCHEME`.
    """
    if mode is not None:
        mode = ReportMode(mode) if not isinstance(mode, ReportMode) else mode
        return _BY_MODE[mode]
    cols = data_df.columns
    for scheme in SPLIT_SCHEMES:
        if scheme is NONE_SCHEME:
            continue
        if scheme.detect(cols):
            return scheme
    return NONE_SCHEME


def resolve_scheme_and_add_columns(
    data_df: pd.DataFrame,
    adapter_version: str | None = None,
    categorical_features_names: list[str] | None = None,
) -> tuple[pd.DataFrame, SplitScheme]:
    """Detect the active scheme, add the columns it needs, and add the ordered ``read_group``.

    ``read_group`` is written from the scheme's *display* variant, giving a uniform N-group column
    for plots that read it directly. Returns ``(df, scheme)``.
    """
    scheme = resolve_scheme(data_df)
    logger.info("Detected SRSNV split scheme: %s", scheme.mode.value)
    data_df = scheme.add_columns(
        data_df,
        {"adapter_version": adapter_version, "categorical_features_names": categorical_features_names},
    )
    data_df[READ_GROUP] = scheme.display_variant.group_fn(data_df)
    return data_df, scheme
