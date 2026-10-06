"""Unit tests for the paired-end-duplex split scheme (simplex SE / simplex PE / duplex SE / duplex PE).

Covers the pe-duplex grouping driven by the DNN per-image CS-family stats (``cs_n_crossing`` +
``DS``/``mate_present``): scheme detection/priority over the generic DUPLEX scheme, the
simplex-SE / simplex-PE / duplex-SE / duplex-PE assignment (a single-read family is simplex SE), and
the legacy fallback when the CS stats are absent.
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
    PE_DUPLEX_GROUP_SIMPLEX_PE,
    PE_DUPLEX_GROUP_SIMPLEX_SE,
    PE_DUPLEX_GROUPS,
    READ_GROUP,
    ReportMode,
)


def _pe_duplex_df() -> pd.DataFrame:
    """One row per outcome: simplex SE (DS!=2, both PE ends on the single strand so fam==2 but only one
    crosses -> cross==1), simplex PE (DS!=2 & cross==2), duplex SE (DS==2 & cross==2), duplex PE
    (DS==2 & cross==4), and a single-read family (cs_family_size==1) which is also a simplex-SE
    observation (one strand, one PE end crossing)."""
    return pd.DataFrame(
        {
            "CHROM": ["chr1"] * 5,
            "POS": [10, 15, 20, 30, 40],
            "RN": ["r_simplex_se", "r_simplex_pe", "r_se", "r_pe", "r_fam1"],
            NF: [2, 2, 2, 2, 1],
            NR: [0, 0, 2, 2, 0],
            DS: [1, 1, 2, 2, 1],
            CS_FAMILY: ["c_simplex_se", "c_simplex_pe", "c_se", "c_pe", "c_fam1"],
            CS_FAMILY_SIZE: [2, 2, 2, 4, 1],
            CS_N_CROSSING: [1, 2, 2, 4, 1],
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


def test_pe_duplex_grouping_four_groups_and_fam1_is_simplex_se():
    frame = _pe_duplex_df()
    out, scheme = resolve_scheme_and_add_columns(frame.copy())
    assert scheme is PE_DUPLEX_SCHEME
    rg = list(out[READ_GROUP])
    # simplex SE / simplex PE / duplex SE / duplex PE; the single-read family (fam==1) is simplex SE too.
    assert rg[0] == PE_DUPLEX_GROUP_SIMPLEX_SE
    assert rg[1] == PE_DUPLEX_GROUP_SIMPLEX_PE
    assert rg[2] == PE_DUPLEX_GROUP_SE
    assert rg[3] == PE_DUPLEX_GROUP_PE
    assert rg[4] == PE_DUPLEX_GROUP_SIMPLEX_SE  # cs_family_size == 1 -> simplex SE (not dropped)
    # group_masks emit all four display groups; nothing is dropped, so simplex SE now has two rows.
    masks = scheme.display_variant.group_masks(out)
    assert set(masks) == set(PE_DUPLEX_GROUPS)
    assert {k: int(v.sum()) for k, v in masks.items()} == {
        PE_DUPLEX_GROUP_SIMPLEX_SE: 2,
        PE_DUPLEX_GROUP_SIMPLEX_PE: 1,
        PE_DUPLEX_GROUP_SE: 1,
        PE_DUPLEX_GROUP_PE: 1,
    }


def test_duplex_se_boundary_three_crossing_is_pe():
    # cs_n_crossing >= 3 is duplex PE (both PE ends); exactly 2 is duplex SE.
    frame = _pe_duplex_df()
    frame.loc[2, CS_N_CROSSING] = 3  # the duplex SE row (DS==2)
    out, _ = resolve_scheme_and_add_columns(frame.copy())
    assert out[READ_GROUP].iloc[2] == PE_DUPLEX_GROUP_PE


def test_simplex_boundary_two_crossing_is_simplex_pe():
    # single strand (DS!=2): 1 crossing read -> simplex SE, 2 -> simplex PE.
    frame = _pe_duplex_df()
    frame.loc[0, CS_N_CROSSING] = 2  # the simplex SE row -> now both ends cross
    out, _ = resolve_scheme_and_add_columns(frame.copy())
    assert out[READ_GROUP].iloc[0] == PE_DUPLEX_GROUP_SIMPLEX_PE
