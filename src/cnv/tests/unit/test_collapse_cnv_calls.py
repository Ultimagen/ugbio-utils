import pysam
import pytest
from ugbio_cnv import collapse_cnv_calls as cc


def _write_vcf(path, calls):
    """calls: (id, start0, end, svtype, filter, qual, source, cn_mean, ci)"""
    header = pysam.VariantHeader()
    header.add_line("##contig=<ID=chr1,length=1000000>")
    header.add_line('##FILTER=<ID=LOW_SCORE,Description="low score">')
    for line in (
        '##INFO=<ID=END,Number=1,Type=Integer,Description="End">',
        '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type">',
        '##INFO=<ID=SVLEN,Number=.,Type=Integer,Description="Length">',
        '##INFO=<ID=CIPOS,Number=2,Type=Integer,Description="CI start">',
        '##INFO=<ID=CIEND,Number=2,Type=Integer,Description="CI end">',
        '##INFO=<ID=CNV_SOURCE,Number=.,Type=String,Description="Source">',
        '##INFO=<ID=CN_MEAN,Number=1,Type=Float,Description="CN">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
    ):
        header.add_line(line)
    header.add_sample("S")
    with pysam.VariantFile(str(path), "wz", header=header) as out:
        for cid, start, end, svtype, flt, qual, source, cn, ci in sorted(calls, key=lambda c: c[1]):
            rec = header.new_record(contig="chr1", start=start, stop=end, alleles=("N", f"<{svtype}>"), id=cid)
            rec.qual = qual
            rec.filter.add(flt)
            rec.info["SVTYPE"], rec.info["SVLEN"] = svtype, end - start
            rec.info["CIPOS"], rec.info["CIEND"] = (-ci, ci), (-ci, ci)
            rec.info["CNV_SOURCE"], rec.info["CN_MEAN"] = (source,), cn
            rec.samples["S"]["GT"] = (0, 1)
            out.write(rec)
    pysam.tabix_index(str(path), preset="vcf", force=True)
    return str(path)


def _calls(vcf):
    with pysam.VariantFile(vcf) as v:
        return {
            r.id: (r.start, r.stop, r.info["SVTYPE"], next(iter(r.filter.keys())), r.info.get("CNV_ID"), r.info)
            for r in v
        }


class TestMergedBreakpoints:
    def test_precise_breakpoint_wins_within_the_window(self):
        members = [
            {"start": 1000, "end": 9000, "cipos": (-500, 500), "ciend": (-500, 500)},
            {"start": 1800, "end": 8800, "cipos": (-10, 10), "ciend": (-10, 10)},
        ]
        assert cc.merged_breakpoints(members, window=2500) == (1800, 8800, (-10, 10), (-10, 10))

    def test_end_never_before_start(self):
        # the tightest CIPOS start (2000) lies after the tightest CIEND end (1500): the end is taken after the start
        members = [
            {"start": 0, "end": 4000, "cipos": (-500, 500), "ciend": (-500, 500)},
            {"start": 2000, "end": 3500, "cipos": (-10, 10), "ciend": (-200, 200)},
            {"start": 500, "end": 1500, "cipos": (-200, 200), "ciend": (-10, 10)},
        ]
        start, end, _, _ = cc.merged_breakpoints(members, window=2500)
        assert (start, end) == (2000, 3500)

    def test_outer_edges_kept_beyond_the_window(self):
        members = [
            {"start": 1000, "end": 20000, "cipos": (-500, 500), "ciend": (-500, 500)},
            {"start": 5000, "end": 9000, "cipos": (-10, 10), "ciend": (-10, 10)},
        ]
        assert cc.merged_breakpoints(members, window=2500) == (1000, 20000, (-500, 500), (-500, 500))


@pytest.fixture
def collapsed(tmp_path):
    calls = [
        # a depth DEL and a split-read DEL on one event: one call, the split-read breakpoints
        ("DEPTH_1", 10000, 20000, "DEL", "PASS", 40.0, "depth", 1.0, 500),
        ("SR_1", 10400, 19600, "DEL", "PASS", 35.0, "split_read", 1.2, 10),
        # a DUP overlapping that DEL: not merged with it
        ("DEPTH_2", 15000, 30000, "DUP", "PASS", 33.0, "depth", 3.0, 500),
        # a chain: A overlaps B, B overlaps C, A and C do not touch; one call covering all three
        ("A", 50000, 52000, "DEL", "PASS", 30.0, "depth", 1.0, 500),
        ("B", 51500, 60000, "DEL", "PASS", 45.0, "depth", 0.0, 500),
        ("C", 59000, 62000, "DEL", "PASS", 31.0, "depth", 1.0, 500),
        # LOW_SCORE calls: a DEL and a DUP inside the PASS DEL (both dropped, whatever their type), a DEL that only
        # half-touches the PASS DEL B (kept: < 50% inside), and one on its own (kept)
        ("LOW_1", 12000, 14000, "DEL", "LOW_SCORE", 20.0, "depth", 1.0, 500),
        ("LOW_DUP", 11000, 13000, "DUP", "LOW_SCORE", 20.0, "pair", 3.0, 200),
        ("LOW_EDGE", 61000, 66000, "DEL", "LOW_SCORE", 20.0, "depth", 1.0, 500),
        ("LOW_2", 80000, 85000, "DEL", "LOW_SCORE", 20.0, "depth", 1.0, 500),
    ]
    vcf = _write_vcf(tmp_path / "in.vcf.gz", calls)
    out = str(tmp_path / "out.vcf.gz")
    n = cc.collapse_cnv_calls(vcf, out)
    return n, _calls(out)


class TestCollapseCnvCalls:
    def test_same_type_overlap_becomes_one_call(self, collapsed):
        n, calls = collapsed
        assert "SR_1" not in calls
        start, end, svtype, flt, cnv_id, info = calls["DEPTH_1"]  # the higher QUAL is kept
        assert (start, end, svtype, flt) == (10400, 19600, "DEL", "PASS")  # with the precise split-read breakpoints
        assert tuple(cnv_id) == ("SR_1",)
        assert set(info["CNV_SOURCE"]) == {"depth", "split_read"}
        assert info["SVLEN"][0] == 19600 - 10400
        assert info["CIPOS"] == (-10, 10)

    def test_del_and_dup_never_merged(self, collapsed):
        _, calls = collapsed
        assert calls["DEPTH_2"][:4] == (15000, 30000, "DUP", "PASS")

    def test_chain_covers_the_whole_group(self, collapsed):
        n, calls = collapsed
        assert "A" not in calls and "C" not in calls
        start, end, _, _, cnv_id, info = calls["B"]
        assert (start, end) == (50000, 62000)
        assert sorted(cnv_id) == ["A", "C"]
        assert info["CN_MEAN"] == pytest.approx((1.0 * 2000 + 0.0 * 8500 + 1.0 * 3000) / 13500, abs=1e-3)
        assert n == 3  # SR_1, A, C

    def test_genotype_follows_the_merged_copy_number(self, tmp_path):
        # the kept (higher QUAL) deletion is heterozygous, but the merged copy number is below 0.5: GT becomes 1/1
        calls = [
            ("HET", 10000, 12000, "DEL", "PASS", 40.0, "split_read", 1.0, 10),
            ("HOM", 10500, 20500, "DEL", "PASS", 30.0, "depth", 0.0, 500),
        ]
        out = str(tmp_path / "out.vcf.gz")
        cc.collapse_cnv_calls(_write_vcf(tmp_path / "in.vcf.gz", calls), out)
        with pysam.VariantFile(out) as v:
            (rec,) = list(v)
        assert rec.id == "HET" and rec.info["CN_MEAN"] < 0.5
        assert rec.samples["S"]["GT"] == (1, 1)

    def test_filtered_calls_inside_a_pass_call_dropped(self, collapsed):
        _, calls = collapsed
        assert "LOW_1" not in calls  # a DEL inside the PASS DEL
        assert "LOW_DUP" not in calls  # a DUP inside the PASS DEL: dropped too, the type does not matter

    def test_other_filtered_calls_kept_unchanged(self, collapsed):
        _, calls = collapsed
        assert calls["LOW_EDGE"][:4] == (61000, 66000, "DEL", "LOW_SCORE")  # 1 kb of 5 kb inside the merged B
        assert calls["LOW_2"][:4] == (80000, 85000, "DEL", "LOW_SCORE")
        assert calls["LOW_2"][4] is None

    def test_at_most_two_pass_calls_at_any_position(self, tmp_path):
        # stacked PASS calls: three DELs and two DUPs over the same region; after the collapse one DEL and one DUP
        calls = [
            ("D1", 10000, 20000, "DEL", "PASS", 40.0, "depth", 1.0, 500),
            ("D2", 12000, 18000, "DEL", "PASS", 35.0, "split_read", 1.0, 10),
            ("D3", 15000, 25000, "DEL", "PASS", 30.0, "pair", 1.0, 200),
            ("U1", 11000, 30000, "DUP", "PASS", 33.0, "depth", 3.0, 500),
            ("U2", 14000, 16000, "DUP", "PASS", 31.0, "split_read", 3.0, 10),
        ]
        out = str(tmp_path / "out.vcf.gz")
        cc.collapse_cnv_calls(_write_vcf(tmp_path / "in.vcf.gz", calls), out)
        passing = [(s, e) for s, e, _, flt, _, _ in _calls(out).values() if flt == "PASS"]
        assert len(passing) == 2  # noqa: PLR2004
        positions = {p for s, e in passing for p in (s, e - 1)}
        assert max(sum(s <= p < e for s, e in passing) for p in positions) <= 2  # noqa: PLR2004

    def test_filtered_calls_all_kept_when_disabled(self, tmp_path):
        calls = [
            ("P", 10000, 20000, "DEL", "PASS", 40.0, "depth", 1.0, 500),
            ("L", 12000, 14000, "DEL", "LOW_SCORE", 20.0, "depth", 1.0, 500),
        ]
        out = str(tmp_path / "out.vcf.gz")
        cc.collapse_cnv_calls(_write_vcf(tmp_path / "in.vcf.gz", calls), out, filtered_inside_frac=0)
        assert set(_calls(out)) == {"P", "L"}


@pytest.mark.parametrize(("start", "end", "expected"), [(0, 100, 1.0), (50, 150, 0.5), (200, 300, 0.0), (90, 110, 0.5)])
def test_inside_fraction(start, end, expected):
    import numpy as np

    assert cc.inside_fraction(start, end, np.array([0, 100]), np.array([100, 100])) == pytest.approx(expected)
