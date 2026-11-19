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
    MIXED_SCHEME,
    NONE_SCHEME,
    PE_DUPLEX_SCHEME,
    SplitScheme,
    SplitVariant,
    h5_key,
    resolve_scheme,
    resolve_scheme_and_add_columns,
)
from ugbio_srsnv.srsnv_utils import (
    DX,
    FS,
    IS_CONSENSUS,
    IS_MIXED,
    MATE_PRESENT,
    PE_DUPLEX_GROUP_PE,
    PE_DUPLEX_GROUP_SE,
    PE_DUPLEX_GROUP_SIMPLEX_PE,
    PE_DUPLEX_GROUP_SIMPLEX_SE,
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

    def test_no_cs_stats_resolves_as_before(self):
        # Regression: without the PE-duplex CS-family stats, detection is unchanged.
        assert resolve_scheme(pd.DataFrame({"st": ["MIXED"], "et": ["MIXED"]})) is MIXED_SCHEME
        assert resolve_scheme(pd.DataFrame({FS: [0, 1], RS: [1, 1]})) is CONSENSUS_SCHEME
        assert resolve_scheme(pd.DataFrame({"MQUAL": [1.0, 2.0]})) is NONE_SCHEME

    def test_pe_duplex_mate_present_from_dx(self):
        # PE-duplex run (CS-family stats present). A molecule is duplex iff dx==1 (coverage-based; dx absent ->
        # not duplex); mate_present is derived from dx, NOT from DS or from whether both strands appear.
        data_df = pd.DataFrame(
            {
                "CS": ["X", "X", "Y", "Z", "Z"],
                "nf": [3, 0, 2, 1, 0],
                "nr": [0, 5, 0, 0, 1],
                DX: [1, 1, None, 1, None],
                "DS": [4, 4, 2, 2, 2],  # family size on the paired CRAMs: must be ignored
                "cs_family_size": [4, 4, 1, 2, 2],
                "cs_n_crossing": [4, 2, 1, 2, 2],
                "st": ["MIXED"] * 5,
                "et": ["MIXED"] * 5,
            }
        )
        out, scheme = resolve_scheme_and_add_columns(data_df.copy())
        assert scheme is PE_DUPLEX_SCHEME
        assert list(out[MATE_PRESENT].astype(bool)) == [True, True, False, True, False]
        assert list(out[READ_GROUP]) == [
            PE_DUPLEX_GROUP_PE,
            PE_DUPLEX_GROUP_SE,
            PE_DUPLEX_GROUP_SIMPLEX_SE,
            PE_DUPLEX_GROUP_SE,
            PE_DUPLEX_GROUP_SIMPLEX_PE,
        ]
        assert "duplex_concordant" not in out.columns

    def test_nf_nr_without_cs_stats_resolves_to_mixed(self):
        # Single-end duplex is no longer supported: nf/nr consensus counts alone (no PE-duplex CS-family
        # stats) do not select a duplex split; st/et ppmSeq tags -> MIXED, fs/rs -> CONSENSUS.
        data_df = pd.DataFrame({"nf": [3, 0, 2, 0], "nr": [5, 4, 0, 0], "st": ["MIXED"] * 4, "et": ["MIXED"] * 4})
        assert resolve_scheme(data_df) is MIXED_SCHEME
        assert resolve_scheme(pd.DataFrame({MATE_PRESENT: [1, 0], FS: [1, 2], RS: [1, 0]})) is CONSENSUS_SCHEME

    def test_duplex_molecule_mode_is_rejected(self):
        with pytest.raises(ValueError, match="duplex_molecule"):
            resolve_scheme(mode="duplex_molecule")


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
        for scheme in (MIXED_SCHEME, PE_DUPLEX_SCHEME, CONSENSUS_SCHEME, NONE_SCHEME):
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


class TestStartOnlyPpmseq:
    """Start-only ppmSeq libraries (st, no et; BIOIN-3125) keep the mixed/non-mixed start split."""

    def test_start_only_detected_as_mixed(self):
        assert resolve_scheme(pd.DataFrame({"st": ["MIXED", "PLUS"]})) is MIXED_SCHEME
        assert resolve_scheme(pd.DataFrame({"st": ["MIXED"], FS: [1], RS: [1]})) is MIXED_SCHEME

    def test_start_only_keeps_start_split(self):
        data_df = pd.DataFrame({"st": ["MIXED", "PLUS", "MIXED", "MINUS"]})
        out, scheme = resolve_scheme_and_add_columns(data_df, adapter_version="v1")
        assert scheme is MIXED_SCHEME
        assert out["is_mixed_start"].tolist() == [True, False, True, False]
        assert not out["is_mixed_end"].any()
        assert not out[IS_MIXED].any()  # both-ends legacy split collapses (no end tag)
        assert list(out[READ_GROUP].astype(str)) == ["Mixed", "Non-mixed", "Mixed", "Non-mixed"]
        # the display (start-tag) variant still yields both groups
        masks = MIXED_SCHEME.display_variant.group_masks(out)
        assert masks["Mixed"].sum() == 2 and masks["Non-mixed"].sum() == 2
