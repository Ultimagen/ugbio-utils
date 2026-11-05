import numpy as np
import pandas as pd
from ugbio_cnv import pair_candidates as pc
from ugbio_cnv.split_read_candidates import CandidateParams

CP = CandidateParams()


def pairs(rows):
    return pd.DataFrame(rows, columns=["chrom", "left_start", "left_end", "right_start", "right_end", "svtype"])


def test_deletion_is_the_gap_between_the_reads_and_the_mates():
    # five FR pairs across a deletion, two stray pairs elsewhere
    le = [950 + k * 20 for k in range(5)]
    rs = [4200 + k * 15 for k in range(5)]
    p = pairs(
        [("chr1", e - 150, e, s, s + 150, "DEL") for e, s in zip(le, rs, strict=True)]
        + [("chr1", 90000, 90150, 95000, 95150, "DEL"), ("chr2", 100, 250, 4000, 4150, "DEL")]
    )
    c = pc.cluster_pairs(p, pc.PairParams(), CP)
    assert len(c) == 1
    r = c.iloc[0]
    assert (r.chrom, r.svtype, r.n_reads) == ("chr1", "DEL", 5)
    # same rule as breakpoint_snapping.pair_bounds
    assert (r.start, r.end) == (int(np.percentile(le, 90)), int(np.percentile(rs, 10)))


def test_duplication_contains_the_everted_pairs():
    # the duplication ends at the mates' ends (left start + TLEN), not their starts
    p = pairs([("chr1", 10000 + k * 30, 10150 + k * 30, 14550 - k * 20, 14700 - k * 20, "DUP") for k in range(3)])
    c = pc.cluster_pairs(p, pc.PairParams(), CP)
    assert list(c.svtype) == ["DUP"]
    assert (c.start.iloc[0], c.end.iloc[0]) == (10000, 14700)


def test_size_and_pair_limits():
    # deletions are kept from a 300 bp gap (500 bp is required after snapping), duplications from 500 bp
    gap400 = pairs([("chr1", 850, 1000, 1400 + k, 1550 + k, "DEL") for k in range(5)])
    assert len(pc.cluster_pairs(gap400, pc.PairParams(), CP)) == 1
    gap250 = pairs([("chr1", 850, 1000, 1250 + k, 1400 + k, "DEL") for k in range(5)])
    assert pc.cluster_pairs(gap250, pc.PairParams(), CP).empty
    dup400 = pairs([("chr1", 1000 + k, 1150 + k, 1250, 1400, "DUP") for k in range(3)])
    assert pc.cluster_pairs(dup400, pc.PairParams(), CP).empty
    few = pairs([("chr1", 800 + k, 950 + k, 4200, 4350, "DEL") for k in range(2)])
    assert pc.cluster_pairs(few, pc.PairParams(), CP).empty
    assert len(pc.cluster_pairs(few, pc.PairParams(min_pairs=2), CP)) == 1


def test_far_apart_pairs_are_separate_clusters():
    p = pairs([("chr1", 800 + k, 950 + k, 4200, 4350, "DEL") for k in range(3)])
    far = p.assign(left_end=p.left_end + 700, left_start=p.left_start + 700)  # beyond cluster_tol
    c = pc.cluster_pairs(pd.concat([p, far], ignore_index=True), pc.PairParams(), CP)
    assert len(c) == 2


def test_deletion_span_floor_is_the_evidence_step_floor():
    from ugbio_cnv.analyze_cnv_breakpoint_reads import MIN_PAIR_SPAN

    assert pc.PairParams().min_span == MIN_PAIR_SPAN
