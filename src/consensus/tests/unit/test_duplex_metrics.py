from ugbio_consensus import duplex_metrics
from ugbio_consensus.duplex_metrics import DUPLEX, SINGLE_STRAND, SINGLETON, classify_family


def test_classify_duplex():
    assert classify_family(4, 4) == DUPLEX
    assert classify_family(1, 6) == DUPLEX


def test_classify_single_strand():
    assert classify_family(5, 0) == SINGLE_STRAND
    assert classify_family(0, 3) == SINGLE_STRAND


def test_classify_singleton():
    assert classify_family(1, 0) == SINGLETON
    assert classify_family(0, 1) == SINGLETON
    assert classify_family(0, 0) == SINGLETON


def test_merge_intervals_disjoint():
    merged = duplex_metrics._merge_intervals(
        [("chr1", 100, 200), ("chr1", 150, 250), ("chr1", 400, 500), ("chr2", 0, 50)]
    )
    assert merged == [("chr1", 100, 250), ("chr1", 400, 500), ("chr2", 0, 50)]


def test_categories_order():
    assert duplex_metrics.CATEGORIES == (DUPLEX, SINGLE_STRAND, SINGLETON)
