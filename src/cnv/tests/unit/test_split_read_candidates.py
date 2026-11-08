from pathlib import Path

import pandas as pd
import pysam
import pytest
from ugbio_cnv import split_read_candidates as src
from ugbio_cnv.depth_cnv_caller import write_depth_calls


@pytest.fixture
def fasta_index():
    return str(Path(__file__).parent.parent / "resources" / "dummy.fasta.fai")


def depth_calls(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chrom": c,
                "start": s,
                "end": e,
                "direction": t,
                "low_conf": False,
                "cn_mean": 1.0,
                "baseline": 2.0,
                "z": -10.0,
                "n_merged": 1,
                "segdup": 0.0,
                "uniq_frac": 1.0,
            }
            for c, s, e, t in rows
        ]
    )


def candidates(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chrom": c,
                "start": s,
                "end": e,
                "svtype": t,
                "n_reads": n,
                "cn_mean": 1.0,
                "baseline": 2.0,
                "z": -8.0,
                "segdup": 0.0,
                "uniq_frac": 1.0,
            }
            for c, s, e, t, n in rows
        ]
    )


class TestClustering:
    def test_cluster_and_minimum_reads(self):
        j = pd.DataFrame(
            [
                ("chr1", 1000, 3000, "DEL"),
                ("chr1", 1010, 3005, "DEL"),
                ("chr1", 990, 2995, "DEL"),
                ("chr1", 50000, 52000, "DEL"),
                ("chr1", 1000, 3000, "DUP"),
                ("chr1", 1005, 3000, "DUP"),
            ],
            columns=["chrom", "start", "end", "svtype"],
        )
        c = src.cluster_junctions(j, src.CandidateParams()).sort_values("svtype")
        assert list(map(tuple, c.to_numpy())) == [("chr1", 1000, 3000, "DEL", 3), ("chr1", 1002, 3000, "DUP", 2)]

    def test_dedup_keeps_best_supported(self):
        c = pd.DataFrame(
            [("chr1", 1000, 5000, "DEL", 3), ("chr1", 1200, 5100, "DEL", 8), ("chr1", 20000, 25000, "DEL", 2)],
            columns=["chrom", "start", "end", "svtype", "n_reads"],
        )
        out = src.dedup_clusters(c, src.CandidateParams())
        assert sorted(out["n_reads"]) == [2, 8]


class TestMergeWithDepthCalls:
    def test_already_called_candidates_are_not_added(self, tmp_path, fasta_index):
        dvcf = str(tmp_path / "depth.vcf.gz")
        write_depth_calls(depth_calls([("chr1", 10000, 20000, "DEL")]), dvcf, "S1", fasta_index)
        cand = candidates(
            [("chr1", 10200, 19900, "DEL", 5), ("chr1", 50000, 51500, "DEL", 4), ("chr1", 10200, 19900, "DUP", 3)]
        )
        out = str(tmp_path / "cand.vcf.gz")
        assert src.add_candidates_to_calls(dvcf, cand, out, src.CandidateParams()) == 2
        with pysam.VariantFile(out) as v:
            recs = [(r.start, r.stop, r.info["SVTYPE"], r.info["CNV_SOURCE"], r.info.get("SR_READS")) for r in v]
            gts = [r.samples["S1"]["GT"] for r in pysam.VariantFile(out)]
        assert recs == [
            (10000, 20000, "DEL", ("depth",), None),
            (10200, 19900, "DUP", ("split_read",), 3),
            (50000, 51500, "DEL", ("split_read",), 4),
        ]
        assert gts == [(0, 1), (None, 1), (0, 1)]


class TestNestedFilter:
    def test_nested_and_short_candidates_dropped(self, tmp_path, fasta_index):
        dvcf = str(tmp_path / "depth.vcf.gz")
        write_depth_calls(depth_calls([("chr1", 100_000, 900_000, "DEL")]), dvcf, "S1", fasta_index)
        cand = candidates(
            [
                ("chr1", 400_000, 402_000, "DEL", 6),  # inside the depth deletion
                ("chr1", 400_000, 402_000, "DUP", 6),  # inside, but the other type: kept
                ("chr1", 950_000, 950_400, "DEL", 3),  # 400 bp: too short
                ("chr1", 970_000, 975_000, "DEL", 3),  # independent: kept
            ]
        )
        cvcf, out = str(tmp_path / "cand.vcf.gz"), str(tmp_path / "filtered.vcf.gz")
        src.add_candidates_to_calls(dvcf, cand, cvcf, src.CandidateParams())
        dropped = src.filter_nested_cnv_candidates(cvcf, out)
        assert dropped == {"nested": 1, "short": 1, "duplicate": 0}
        with pysam.VariantFile(out) as v:
            kept = [(r.start, r.info["SVTYPE"]) for r in v]
        assert kept == [(100_000, "DEL"), (400_000, "DUP"), (970_000, "DEL")]

    def test_duplicate_candidates_dropped(self, tmp_path, fasta_index):
        """Candidates snapped onto the same event: split-read kept first, then the pair candidate with most pairs."""
        dvcf = str(tmp_path / "depth.vcf.gz")
        write_depth_calls(depth_calls([("chr1", 10_000, 12_000, "DEL")]), dvcf, "S1", fasta_index)
        sr = str(tmp_path / "sr.vcf.gz")
        src.add_candidates_to_calls(dvcf, candidates([("chr1", 200_000, 210_000, "DEL", 2)]), sr, src.CandidateParams())
        pairs = candidates(
            [
                ("chr1", 200_100, 210_050, "DEL", 9),  # same event as the split-read candidate: dropped
                ("chr1", 300_000, 320_000, "DEL", 3),  # same event as the next one, fewer pairs: dropped
                ("chr1", 300_500, 320_500, "DEL", 8),
                ("chr1", 300_000, 320_000, "DUP", 3),  # the other type: kept
                ("chr1", 305_000, 309_000, "DEL", 4),  # inside the 20 kb candidate, but not >= 50% of it: kept
            ]
        )
        cvcf, out = str(tmp_path / "cand.vcf.gz"), str(tmp_path / "filtered.vcf.gz")
        p = src.CandidateParams(min_overlap=1.1)  # add all of them (the "already called" check is not tested here)
        src.add_candidates_to_calls(
            sr, pairs, cvcf, p, source=src.CNV_SOURCE_PAIR, id_prefix="PAIR", count_tag="PAIR_READS"
        )
        dropped = src.filter_nested_cnv_candidates(cvcf, out)
        assert dropped == {"nested": 0, "short": 0, "duplicate": 2}
        with pysam.VariantFile(out) as v:
            kept = [(r.start, r.info["SVTYPE"], r.info["CNV_SOURCE"]) for r in v]
        assert kept == [
            (10_000, "DEL", ("depth",)),
            (200_000, "DEL", ("split_read",)),
            (300_000, "DUP", ("pair",)),
            (300_500, "DEL", ("pair",)),
            (305_000, "DEL", ("pair",)),
        ]
