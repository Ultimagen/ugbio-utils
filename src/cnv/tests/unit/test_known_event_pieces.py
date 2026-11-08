import numpy as np
import pandas as pd
from ugbio_cnv import depth_cnv_caller as dc
from ugbio_cnv import known_event_pieces as kep

EVENT = ("chr1", 1_000_000, 1_500_000, "DUP")


def synthetic_bins(segdup_region=(1_200_000, 1_300_000)) -> pd.DataFrame:
    """2 Mb of chr1 at CN 2 with a CN 3 event; a segdup stretch inside the event."""
    rng = np.random.default_rng(0)
    start = np.arange(0, 2_000_000, dc.BIN_SIZE)
    cn = np.where((start >= EVENT[1]) & (start < EVENT[2]), 3.0, 2.0) + rng.normal(0, 0.2, len(start))
    segdup = ((start >= segdup_region[0]) & (start < segdup_region[1])).astype(float)
    return pd.DataFrame(
        {
            "chrom": "chr1",
            "start": start,
            "end": start + dc.BIN_SIZE,
            "callable": True,
            "cn": cn,
            "baseline": 2.0,
            "lowmap": 0.0,
            "segdup": segdup,
            "uniq_frac": 1.0,
            "n_frac": 0.0,
        }
    )


def test_pieces_lie_inside_the_event_in_clean_sequence():
    p = dc.Params()
    bins = synthetic_bins()
    units = kep.unit_table(bins, p)
    pieces = kep.cut_pieces(bins, units, EVENT, 6, 10_000, 50_000, np.random.default_rng(1), p)
    assert len(pieces) == 6
    assert (pieces["start"] >= EVENT[1] - dc.BIN_SIZE).all()
    assert (pieces["end"] <= EVENT[2]).all()
    assert pieces["end"].sub(pieces["start"]).between(10_000 - dc.BIN_SIZE, 50_000).all()
    assert (pieces["direction"] == "DUP").all()
    assert pieces["cn_mean"].between(2.8, 3.2).all()
    assert (pieces["bin_segdup"] <= 0.1).all()  # the segdup stretch is avoided
    # no two pieces overlap (max_overlap 0)
    iv = pieces.sort_values("start")
    assert (iv["start"].to_numpy()[1:] >= iv["end"].to_numpy()[:-1]).all()


def test_pieces_are_reproducible_with_the_seed():
    p = dc.Params()
    bins = synthetic_bins()
    units = kep.unit_table(bins, p)
    a = kep.cut_pieces(bins, units, EVENT, 4, 10_000, 50_000, np.random.default_rng(7), p)
    b = kep.cut_pieces(bins, units, EVENT, 4, 10_000, 50_000, np.random.default_rng(7), p)
    pd.testing.assert_frame_equal(a, b)


def test_no_pieces_when_the_depth_goes_the_other_way():
    p = dc.Params()
    bins = synthetic_bins()
    units = kep.unit_table(bins, p)
    pieces = kep.cut_pieces(
        bins, units, (*EVENT[:3], "DEL"), 3, 10_000, 50_000, np.random.default_rng(1), p, max_tries=500
    )
    assert pieces.empty
