"""Recalibration cell helpers: nearest populated hmer run-length bin (BIOIN-3239)."""

from ugbio_featuremap.featuremap_utils import CELL_SNV, HMER_RUN_BINS, hmer_bin_cell, nearest_bin_cell


def _cells(*labels):
    return {hmer_bin_cell(label) for label in labels}


def test_nearest_bin_prefers_self():
    assert nearest_bin_cell(hmer_bin_cell("L2"), _cells("L1", "L2", "L3-4")) == hmer_bin_cell("L2")


def test_longest_bin_borrows_its_neighbour():
    have = _cells("L1", "L2", "L3-4", "L5-6", "L7-9")
    assert nearest_bin_cell(hmer_bin_cell("L10+"), have) == hmer_bin_cell("L7-9")


def test_tie_goes_to_shorter_run():
    assert nearest_bin_cell(hmer_bin_cell("L3-4"), _cells("L2", "L5-6")) == hmer_bin_cell("L2")


def test_skips_missing_bins():
    assert nearest_bin_cell(hmer_bin_cell("L10+"), _cells("L1", "L3-4")) == hmer_bin_cell("L3-4")


def test_none_for_non_bin_or_empty():
    assert nearest_bin_cell(CELL_SNV, _cells(*[b[0] for b in HMER_RUN_BINS])) is None
    assert nearest_bin_cell(hmer_bin_cell("L2"), set()) is None
    assert nearest_bin_cell(hmer_bin_cell("L99"), _cells("L1")) is None
