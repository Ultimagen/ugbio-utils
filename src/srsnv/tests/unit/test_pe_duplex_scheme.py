"""Unit tests for the paired-end-duplex split scheme (SSC / duplex SE / duplex PE).

Covers the pe-duplex grouping driven by the DNN per-image CS-family stats (``cs_n_crossing`` +
``DS``/``mate_present``): scheme detection/priority over the generic DUPLEX scheme, the SSC /
duplex-SE / duplex-PE assignment, singleton exclusion, and the legacy fallback when the CS stats
are absent.
"""

from __future__ import annotations

import pandas as pd
from ugbio_srsnv.split_scheme import (
    DUPLEX_SCHEME,
    PE_DUPLEX_SCHEME,
    resolve_scheme,
    resolve_scheme_and_add_columns,
)
from ugbio_srsnv.srsnv_utils import (
    CS_FAMILY,
    CS_FAMILY_SIZE,
    CS_N_CROSSING,
    DS,
    NF,
    NR,
    PE_DUPLEX_GROUP_PE,
    PE_DUPLEX_GROUP_SE,
    PE_DUPLEX_GROUP_SSC,
    PE_DUPLEX_GROUPS,
    READ_GROUP,
    ReportMode,
)


def _pe_duplex_df() -> pd.DataFrame:
    """One row per group: SSC (DS!=2), duplex SE (DS==2 & cross==2), duplex PE (DS==2 & cross==4),
    and a singleton (family size 1) that must be excluded from the display groups."""
    return pd.DataFrame(
        {
            "CHROM": ["chr1"] * 4,
            "POS": [10, 20, 30, 40],
            "RN": ["r_ssc", "r_se", "r_pe", "r_single"],
            NF: [3, 2, 2, 0],
            NR: [0, 2, 2, 0],
            DS: [1, 2, 2, 0],
            CS_FAMILY: ["c_ssc", "c_se", "c_pe", "c_single"],
            CS_FAMILY_SIZE: [2, 2, 4, 1],
            CS_N_CROSSING: [2, 2, 4, 1],
        }
    )


def test_pe_duplex_scheme_detected_and_wins_over_duplex():
    frame = _pe_duplex_df()
    # cs_n_crossing present -> pe-duplex scheme; even though nf/nr also match the generic DUPLEX scheme.
    assert resolve_scheme(frame) is PE_DUPLEX_SCHEME
    assert PE_DUPLEX_SCHEME.mode is ReportMode.PE_DUPLEX


def test_pe_duplex_scheme_falls_back_to_duplex_without_cs_stats():
    frame = _pe_duplex_df().drop(columns=[CS_FAMILY_SIZE, CS_N_CROSSING])
    # no CS stats -> the generic duplex scheme (nf/nr) applies, unchanged.
    assert resolve_scheme(frame) is DUPLEX_SCHEME


def test_pe_duplex_grouping_ssc_se_pe_and_singleton_excluded():
    frame = _pe_duplex_df()
    out, scheme = resolve_scheme_and_add_columns(frame.copy())
    assert scheme is PE_DUPLEX_SCHEME
    rg = list(out[READ_GROUP])
    # SSC / duplex SE / duplex PE for the first three rows; singleton -> NaN (dropped from display groups).
    assert rg[0] == PE_DUPLEX_GROUP_SSC
    assert rg[1] == PE_DUPLEX_GROUP_SE
    assert rg[2] == PE_DUPLEX_GROUP_PE
    assert pd.isna(rg[3])
    # group_masks exclude the singleton and only emit the three display groups.
    masks = scheme.display_variant.group_masks(out)
    assert set(masks) == set(PE_DUPLEX_GROUPS)
    assert {k: int(v.sum()) for k, v in masks.items()} == {
        PE_DUPLEX_GROUP_SSC: 1,
        PE_DUPLEX_GROUP_SE: 1,
        PE_DUPLEX_GROUP_PE: 1,
    }


def test_duplex_se_boundary_three_crossing_is_pe():
    # cs_n_crossing >= 3 is duplex PE (both PE ends); exactly 2 is duplex SE.
    frame = _pe_duplex_df()
    frame.loc[1, CS_N_CROSSING] = 3  # was the SE row
    out, _ = resolve_scheme_and_add_columns(frame.copy())
    assert out[READ_GROUP].iloc[1] == PE_DUPLEX_GROUP_PE
