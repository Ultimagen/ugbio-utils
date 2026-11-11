"""Edge cases of the pair-dx helpers in srsnv_utils / srsnv_training not exercised elsewhere.

- add_pe_duplex_columns_to_featuremap_df / _assign_pe_duplex_group: missing dx, missing cs_n_crossing,
  non-numeric values.
- weighted_survival: empty input, NaN handling, weights.
- opportunity counters: empty runs, windows that fetch nothing, CLI parsing of srsnv_training.
"""

import sys

import numpy as np
import pandas as pd
import pytest
from ugbio_srsnv import srsnv_training
from ugbio_srsnv.srsnv_training import _homopolymer_hist_chunked, _homopolymer_runs, _seq_codes
from ugbio_srsnv.srsnv_utils import (
    CS_N_CROSSING,
    DX,
    MATE_PRESENT,
    PE_DUPLEX_GROUP,
    PE_DUPLEX_GROUP_PE,
    PE_DUPLEX_GROUP_SE,
    PE_DUPLEX_GROUP_SIMPLEX_PE,
    PE_DUPLEX_GROUP_SIMPLEX_SE,
    _assign_pe_duplex_group,
    add_pe_duplex_columns_to_featuremap_df,
    weighted_survival,
)

# ---------------------------------------------------------------- pe_duplex_group


def test_pe_duplex_groups_from_dx_and_crossing():
    frame = pd.DataFrame({DX: [None, None, 1, 1, 1, "1"], CS_N_CROSSING: [1, 2, 2, 3, 4, 0]})
    out = add_pe_duplex_columns_to_featuremap_df(frame)
    assert out[MATE_PRESENT].tolist() == [False, False, True, True, True, True]  # "1" coerced to numeric
    assert out[PE_DUPLEX_GROUP].tolist() == [
        PE_DUPLEX_GROUP_SIMPLEX_SE,
        PE_DUPLEX_GROUP_SIMPLEX_PE,
        PE_DUPLEX_GROUP_SE,
        PE_DUPLEX_GROUP_PE,
        PE_DUPLEX_GROUP_PE,
        PE_DUPLEX_GROUP_SE,
    ]


def test_pe_duplex_columns_without_dx_mark_nothing_duplex(caplog):
    frame = pd.DataFrame({CS_N_CROSSING: [1, 2, 5]})
    with caplog.at_level("WARNING"):
        out = add_pe_duplex_columns_to_featuremap_df(frame)
    assert not out[MATE_PRESENT].any()
    # without dx every molecule is simplex; crossing count still splits SE / PE
    assert out[PE_DUPLEX_GROUP].tolist() == [
        PE_DUPLEX_GROUP_SIMPLEX_SE,
        PE_DUPLEX_GROUP_SIMPLEX_PE,
        PE_DUPLEX_GROUP_SIMPLEX_PE,
    ]


def test_pe_duplex_group_noop_without_crossing_stat():
    frame = pd.DataFrame({DX: [1, None]})
    out = add_pe_duplex_columns_to_featuremap_df(frame)
    assert out[MATE_PRESENT].tolist() == [True, False]
    assert PE_DUPLEX_GROUP not in out.columns


def test_assign_pe_duplex_group_without_mate_column_and_bad_values():
    # no mate_present column -> all simplex; a non-numeric / missing crossing count is treated as 0 (SE)
    frame = pd.DataFrame({CS_N_CROSSING: ["x", None, 3]})
    _assign_pe_duplex_group(frame)
    assert frame[PE_DUPLEX_GROUP].tolist() == [
        PE_DUPLEX_GROUP_SIMPLEX_SE,
        PE_DUPLEX_GROUP_SIMPLEX_SE,
        PE_DUPLEX_GROUP_SIMPLEX_PE,
    ]


# ---------------------------------------------------------------- weighted_survival


def test_weighted_survival_empty_is_nan():
    out = weighted_survival(np.array([]), np.array([]), np.array([0.0, 1.0, 2.0]))
    assert out.shape == (3,)
    assert np.isnan(out).all()


def test_weighted_survival_unit_weights_match_unweighted_with_nan():
    vals = np.array([1.0, 2.0, np.nan, 3.0, 2.0])
    x = np.array([0.0, 1.5, 2.0, 2.5, 3.0, 4.0])
    got = weighted_survival(vals, np.ones_like(vals), x)
    expected = np.array([(vals >= t).mean() for t in x])  # NaN never survives but counts in the denominator
    np.testing.assert_allclose(got, expected)
    assert got[0] == pytest.approx(0.8)


def test_weighted_survival_uses_weights():
    vals = np.array([1.0, 5.0])
    w = np.array([3.0, 1.0])
    np.testing.assert_allclose(weighted_survival(vals, w, np.array([0.0, 2.0, 6.0])), [1.0, 0.25, 0.0])


# ---------------------------------------------------------------- opportunity counters


def test_homopolymer_runs_empty_array():
    bases, lengths = _homopolymer_runs(np.zeros(0, dtype=np.uint8))
    assert bases.size == 0
    assert lengths.size == 0
    assert lengths.dtype == np.int64


def test_homopolymer_hist_skips_windows_that_fetch_nothing():
    # The fetch returns fewer bases than `length` (interval past the contig end): trailing windows are empty and
    # must be skipped without dropping the carried-over run. AAAA CC G TTT -> runs of length 4, 2, 1, 3.
    codes = _seq_codes("AAAACCGTTT")
    exact = _homopolymer_hist_chunked(lambda s, e: codes[s:e], codes.size, chunk_size=3)
    padded = _homopolymer_hist_chunked(lambda s, e: codes[s:e], codes.size + 9, chunk_size=3)
    np.testing.assert_array_equal(exact, padded)
    assert dict(enumerate(exact.tolist())) == {0: 0, 1: 1, 2: 1, 3: 1, 4: 1}


def test_homopolymer_hist_ignores_non_acgt_runs():
    codes = _seq_codes("NNNNAAcc")  # lower-case is upper-cased; N runs are not opportunities
    counts = _homopolymer_hist_chunked(lambda s, e: codes[s:e], codes.size, chunk_size=2)
    assert counts.tolist() == [0, 0, 2]


def test_training_cli_parses_args(monkeypatch):
    argv = [
        "srsnv_training",
        "--positive",
        "p.parquet",
        "--negative",
        "n.parquet",
        "--training-regions",
        "r.interval_list",
        "--output",
        "out",
        "--stats-file",
        "s.json",
        "--k-folds",
        "3",
        "--metadata",
        "a=1",
        "--metadata",
        "b=2",
        "--single-model-split",
        "--holdout-chromosomes",
        "chr20,chr21",
        "--val-fraction",
        "0.2",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = srsnv_training._cli()
    assert args.k_folds == 3
    assert args.metadata == ["a=1", "b=2"]
    assert args.single_model_split is True
    assert args.holdout_chromosomes == "chr20,chr21"
    assert args.val_fraction == pytest.approx(0.2)
    assert args.quality_lut_size == 1000
    assert args.split_hash_key == "RN"
    assert args.max_qual == pytest.approx(100.0)
    assert args.use_gpu is False


def test_training_cli_requires_stats_file(monkeypatch):
    argv = ["srsnv_training", "--positive", "p", "--negative", "n", "--training-regions", "r", "--output", "o"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        srsnv_training._cli()


def test_training_main_builds_and_runs_trainer(monkeypatch):
    seen = {}

    class _FakeTrainer:
        def __init__(self, args):
            seen["args"] = args

        def run(self):
            seen["ran"] = True

    argv = ["srsnv_training", "--positive", "p", "--negative", "n", "--training-regions", "r", "--output", "o"]
    argv += ["--stats-file", "s.json", "--verbose"]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(srsnv_training, "SRSNVTrainer", _FakeTrainer)
    level = srsnv_training.logger.level
    try:
        srsnv_training.main()
    finally:
        srsnv_training.logger.setLevel(level)
    assert seen["ran"] is True
    assert seen["args"].positive == "p"
    assert seen["args"].verbose is True
