"""Unit tests for the SRSNV split-scheme "recipe" module (ugbio_srsnv.split_scheme).

Covers: scheme detection from columns, per-scheme add_columns, the ordered group_fns, variant
h5-key composition, and the extensibility contract (a new scheme registered in SPLIT_SCHEMES is
picked up by resolve_scheme without touching any report/figure code).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ugbio_srsnv import split_scheme as ss
from ugbio_srsnv.split_scheme import (
    CONSENSUS_SCHEME,
    DUPLEX_SCHEME,
    MIXED_SCHEME,
    NONE_SCHEME,
    SplitScheme,
    SplitVariant,
    h5_key,
    resolve_scheme,
    resolve_scheme_and_add_columns,
)
from ugbio_srsnv.srsnv_utils import (
    DUPLEX_MOL_PAIRED,
    DUPLEX_MOL_SINGLE_STRAND,
    DUPLEX_MOL_SINGLETON,
    FS,
    IS_CONSENSUS,
    IS_MIXED,
    MATE_PRESENT,
    READ_GROUP,
    RS,
    ReportMode,
)

# ──────────────────────────── detection ────────────────────────────


class TestResolveScheme:
    def test_mixed_from_st_et(self):
        data_df = pd.DataFrame({"st": ["MIXED", "PLUS"], "et": ["MIXED", "MINUS"]})
        assert resolve_scheme(data_df) is MIXED_SCHEME

    def test_mixed_from_v5_tags(self):
        data_df = pd.DataFrame({"as": [1, 2], "ae": [1, 0], "ts": [0, 1], "te": [1, 1]})
        assert resolve_scheme(data_df) is MIXED_SCHEME

    def test_consensus_from_fs_rs(self):
        data_df = pd.DataFrame({FS: [0, 1, 2], RS: [1, 1, 0]})
        assert resolve_scheme(data_df) is CONSENSUS_SCHEME

    def test_none_when_no_split_columns(self):
        data_df = pd.DataFrame({"MQUAL": [10.0, 20.0], "label": [True, False]})
        assert resolve_scheme(data_df) is NONE_SCHEME

    def test_ppmseq_tags_take_priority_over_fs_rs(self):
        data_df = pd.DataFrame({"st": ["MIXED", "PLUS"], "et": ["MIXED", "MINUS"], FS: [1, 2], RS: [1, 0]})
        assert resolve_scheme(data_df) is MIXED_SCHEME

    def test_resolve_by_mode_value(self):
        assert resolve_scheme(mode="mixed") is MIXED_SCHEME
        assert resolve_scheme(mode="consensus") is CONSENSUS_SCHEME
        assert resolve_scheme(mode="none") is NONE_SCHEME

    def test_resolve_by_mode_enum(self):
        assert resolve_scheme(mode=ReportMode.CONSENSUS) is CONSENSUS_SCHEME

    def test_duplex_from_mate_present(self):
        data_df = pd.DataFrame({MATE_PRESENT: [1, 0, 1]})
        assert resolve_scheme(data_df) is DUPLEX_SCHEME

    def test_duplex_takes_priority_over_fs_rs(self):
        # A duplex featuremap may also carry per-read fs/rs; the per-molecule split must win.
        data_df = pd.DataFrame({MATE_PRESENT: [1, 0], FS: [1, 2], RS: [1, 0]})
        assert resolve_scheme(data_df) is DUPLEX_SCHEME

    def test_mate_present_takes_priority_over_ppmseq_tags(self):
        # A duplex run has BOTH st/et (ppmSeq) AND mate_present; the per-molecule duplex split is the
        # point of a duplex run, so DUPLEX must win over MIXED. (Only duplex runs emit mate_present, so
        # non-duplex ppmSeq runs still resolve to MIXED — see test above.)
        data_df = pd.DataFrame({"st": ["MIXED", "PLUS"], "et": ["MIXED", "MINUS"], MATE_PRESENT: [1, 0]})
        assert resolve_scheme(data_df) is DUPLEX_SCHEME

    def test_resolve_duplex_by_mode(self):
        assert resolve_scheme(mode="duplex_molecule") is DUPLEX_SCHEME
        assert resolve_scheme(mode=ReportMode.DUPLEX) is DUPLEX_SCHEME

    def test_no_mate_present_still_resolves_as_before(self):
        # Regression: without mate_present, detection is unchanged.
        assert resolve_scheme(pd.DataFrame({"st": ["MIXED"], "et": ["MIXED"]})) is MIXED_SCHEME
        assert resolve_scheme(pd.DataFrame({FS: [0, 1], RS: [1, 1]})) is CONSENSUS_SCHEME
        assert resolve_scheme(pd.DataFrame({"MQUAL": [1.0, 2.0]})) is NONE_SCHEME


# ──────────────────────────── h5 keys / variants ────────────────────────────


class TestVariantsAndKeys:
    def test_mixed_has_two_variants_legacy_first(self):
        assert len(MIXED_SCHEME.variants) == 2
        # variants[0] is the legacy/base-key variant (empty suffix)
        assert MIXED_SCHEME.legacy_variant.suffix == ""
        assert MIXED_SCHEME.display_variant.suffix == "mixed_start"

    def test_consensus_single_display_variant(self):
        assert len(CONSENSUS_SCHEME.variants) == 1
        assert CONSENSUS_SCHEME.display_variant.suffix == "strand_support"
        assert CONSENSUS_SCHEME.display_suffix == "strand_support"

    def test_h5_key_empty_suffix_is_bare_base(self):
        assert h5_key("run_info_table", MIXED_SCHEME.legacy_variant) == "run_info_table"

    def test_h5_key_nonempty_suffix(self):
        assert h5_key("run_info_table", MIXED_SCHEME.display_variant) == "run_info_table_mixed_start"
        assert h5_key("run_info_table", CONSENSUS_SCHEME.display_variant) == "run_info_table_strand_support"

    def test_exactly_one_display_variant_per_scheme(self):
        for scheme in (MIXED_SCHEME, DUPLEX_SCHEME, CONSENSUS_SCHEME, NONE_SCHEME):
            assert sum(v.is_display for v in scheme.variants) == 1


# ──────────────────────────── group functions ────────────────────────────


class TestGroupFunctions:
    def test_consensus_groups_definition(self):
        # single: fs+rs<=1 ; one strand: >=2 total & exactly one of fs/rs == 0 ; duplex: fs>=1 & rs>=1
        data_df = pd.DataFrame({FS: [0, 0, 3, 1, 2], RS: [0, 1, 0, 1, 3]})
        groups = np.asarray(CONSENSUS_SCHEME.display_variant.group_fn(data_df))
        assert list(groups) == [
            "single read",  # 0+0
            "single read",  # 0+1
            "consensus, one strand",  # 3+0
            "consensus, duplex",  # 1+1
            "consensus, duplex",  # 2+3
        ]

    def test_consensus_groups_are_ordered_categorical(self):
        data_df = pd.DataFrame({FS: [0, 1], RS: [0, 1]})
        cat = CONSENSUS_SCHEME.display_variant.group_fn(data_df)
        assert isinstance(cat, pd.Categorical)
        assert cat.ordered
        assert list(cat.categories) == ["single read", "consensus, one strand", "consensus, duplex"]

    def test_duplex_groups_three_way_with_is_consensus(self):
        # is_consensus present -> 3-way split: mate_present wins -> paired; else is_consensus
        # truthy -> single-strand consensus; else -> singleton.
        data_df = pd.DataFrame(
            {
                MATE_PRESENT: [1, 0, 0, 1],
                IS_CONSENSUS: [1, 1, 0, 0],
            }
        )
        groups = np.asarray(DUPLEX_SCHEME.display_variant.group_fn(data_df))
        assert list(groups) == [
            DUPLEX_MOL_PAIRED,  # mate_present truthy (is_consensus ignored)
            DUPLEX_MOL_SINGLE_STRAND,  # not paired, is_consensus == 1
            DUPLEX_MOL_SINGLETON,  # not paired, is_consensus == 0
            DUPLEX_MOL_PAIRED,  # mate_present truthy wins even if is_consensus == 0
        ]

    def test_duplex_groups_two_way_without_is_consensus(self):
        # Regression: no is_consensus column -> previous 2-way behavior (no singleton group).
        data_df = pd.DataFrame({MATE_PRESENT: [1, 0, 1, 0]})
        groups = np.asarray(DUPLEX_SCHEME.display_variant.group_fn(data_df))
        assert list(groups) == [
            DUPLEX_MOL_PAIRED,
            DUPLEX_MOL_SINGLE_STRAND,
            DUPLEX_MOL_PAIRED,
            DUPLEX_MOL_SINGLE_STRAND,
        ]

    def test_duplex_groups_are_ordered_categorical_ascending_support(self):
        data_df = pd.DataFrame({MATE_PRESENT: [1, 0, 0], IS_CONSENSUS: [1, 1, 0]})
        cat = DUPLEX_SCHEME.display_variant.group_fn(data_df)
        assert isinstance(cat, pd.Categorical)
        assert cat.ordered
        # ascending strand support: singleton -> single-strand consensus -> duplex (paired)
        assert list(cat.categories) == [DUPLEX_MOL_SINGLETON, DUPLEX_MOL_SINGLE_STRAND, DUPLEX_MOL_PAIRED]

    def test_mixed_display_groups_from_is_mixed_start(self):
        data_df = pd.DataFrame({"is_mixed_start": [True, False, True], "is_mixed": [False, False, True]})
        disp = np.asarray(MIXED_SCHEME.display_variant.group_fn(data_df))
        assert list(disp) == ["Mixed", "Non-mixed", "Mixed"]

    def test_mixed_legacy_groups_from_is_mixed(self):
        data_df = pd.DataFrame({"is_mixed_start": [True, False, True], "is_mixed": [False, False, True]})
        legacy = np.asarray(MIXED_SCHEME.legacy_variant.group_fn(data_df))
        assert list(legacy) == ["Non-mixed", "Non-mixed", "Mixed"]

    def test_group_masks_are_boolean_arrays(self):
        data_df = pd.DataFrame({FS: [0, 1, 2], RS: [0, 1, 0]})
        masks = CONSENSUS_SCHEME.display_variant.group_masks(data_df)
        assert set(masks) == {"single read", "consensus, one strand", "consensus, duplex"}
        for m in masks.values():
            assert m.dtype == bool
            assert len(m) == 3
        # exhaustive + mutually exclusive
        stacked = np.vstack(list(masks.values()))
        assert stacked.sum(axis=0).tolist() == [1, 1, 1]


# ──────────────────────────── add_columns / resolve+add ────────────────────────────


class TestAddColumns:
    def test_consensus_adds_is_consensus_and_read_group(self):
        data_df = pd.DataFrame({FS: [0, 1, 2, 3], RS: [1, 1, 0, 2]})
        out, scheme = resolve_scheme_and_add_columns(data_df)
        assert scheme is CONSENSUS_SCHEME
        assert IS_CONSENSUS in out.columns
        expected = ((data_df[FS] >= 1) & (data_df[RS] >= 1)).tolist()
        assert out[IS_CONSENSUS].tolist() == expected
        # read_group is the display variant's grouping
        assert READ_GROUP in out.columns
        np.testing.assert_array_equal((out[READ_GROUP] == "consensus, duplex").to_numpy(), np.array(expected))

    def test_duplex_three_way_coerces_columns_and_adds_read_group(self):
        # is_consensus present -> 3-way split.
        data_df = pd.DataFrame({MATE_PRESENT: [1, 0, 0, 1], IS_CONSENSUS: [1, 1, 0, 0]})
        out, scheme = resolve_scheme_and_add_columns(data_df)
        assert scheme is DUPLEX_SCHEME
        # both columns coerced to bool
        assert out[MATE_PRESENT].dtype == bool
        assert out[MATE_PRESENT].tolist() == [True, False, False, True]
        assert out[IS_CONSENSUS].dtype == bool
        assert out[IS_CONSENSUS].tolist() == [True, True, False, False]
        # read_group is the display variant's grouping
        assert READ_GROUP in out.columns
        assert list(out[READ_GROUP]) == [
            DUPLEX_MOL_PAIRED,
            DUPLEX_MOL_SINGLE_STRAND,
            DUPLEX_MOL_SINGLETON,
            DUPLEX_MOL_PAIRED,
        ]
        assert list(out[READ_GROUP].cat.categories) == [
            DUPLEX_MOL_SINGLETON,
            DUPLEX_MOL_SINGLE_STRAND,
            DUPLEX_MOL_PAIRED,
        ]

    def test_duplex_two_way_without_is_consensus_regression(self):
        # No is_consensus column -> previous 2-way behavior; read_group has no singleton members.
        data_df = pd.DataFrame({MATE_PRESENT: [1, 0, 1, 0]})
        out, scheme = resolve_scheme_and_add_columns(data_df)
        assert scheme is DUPLEX_SCHEME
        assert out[MATE_PRESENT].tolist() == [True, False, True, False]
        assert list(out[READ_GROUP]) == [
            DUPLEX_MOL_PAIRED,
            DUPLEX_MOL_SINGLE_STRAND,
            DUPLEX_MOL_PAIRED,
            DUPLEX_MOL_SINGLE_STRAND,
        ]
        # categories still span all 3 groups (ordered Categorical over DUPLEX_MOL_GROUPS)
        assert list(out[READ_GROUP].cat.categories) == [
            DUPLEX_MOL_SINGLETON,
            DUPLEX_MOL_SINGLE_STRAND,
            DUPLEX_MOL_PAIRED,
        ]

    def test_none_sets_is_consensus_false_and_single_group(self):
        data_df = pd.DataFrame({"MQUAL": [1.0, 2.0, 3.0]})
        out, scheme = resolve_scheme_and_add_columns(data_df)
        assert scheme is NONE_SCHEME
        assert out[IS_CONSENSUS].tolist() == [False, False, False]
        assert list(out[READ_GROUP].unique()) == ["all reads"]

    def test_mixed_adds_is_mixed_columns(self):
        data_df = pd.DataFrame({"st": ["MIXED", "MIXED", "PLUS", "MINUS"], "et": ["MIXED", "PLUS", "PLUS", "MINUS"]})
        out, scheme = resolve_scheme_and_add_columns(data_df, adapter_version="v1")
        assert scheme is MIXED_SCHEME
        assert IS_MIXED in out.columns
        assert READ_GROUP in out.columns


# ──────────────────────────── extensibility contract ────────────────────────────


class TestExtensibility:
    def test_new_scheme_registered_is_detected_without_touching_report(self):
        """A new scheme = one SplitScheme + detector appended to SPLIT_SCHEMES; resolve_scheme
        picks it up. This is the core generality claim of the refactor."""
        marker = "duplex_v2_marker"

        def _dv2_groups(df):
            g = pd.Series("weak", index=df.index, dtype=object)
            g[df[marker] > 0] = "strong"
            return pd.Categorical(g, categories=["weak", "strong"], ordered=True)

        dummy = SplitScheme(
            mode=ReportMode.NONE,  # reuse an enum value; detection is by `detect`, not mode
            detect=lambda cols: marker in cols,
            add_columns=lambda data_df, kw: data_df,  # noqa: ARG005
            variants=(
                SplitVariant(
                    suffix="duplex_v2",
                    group_fn=_dv2_groups,
                    groups=("weak", "strong"),
                    is_display=True,
                ),
            ),
            tag_axis="duplex v2",
        )

        original = ss.SPLIT_SCHEMES
        try:
            # insert before NONE (the fallback) so it can win detection
            ss.SPLIT_SCHEMES = (dummy, *original)
            data_df = pd.DataFrame({marker: [0, 1, 2]})
            assert resolve_scheme(data_df) is dummy
            # and a scheme without the marker still resolves to a built-in
            assert resolve_scheme(pd.DataFrame({FS: [1], RS: [1]})) is CONSENSUS_SCHEME
        finally:
            ss.SPLIT_SCHEMES = original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
