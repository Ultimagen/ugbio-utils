from pathlib import Path

import numpy as np
import pandas as pd
import pysam
import pytest
from ugbio_cnv import depth_cnv_caller as dc


@pytest.fixture
def fasta_index():
    return str(Path(__file__).parent.parent / "resources" / "dummy.fasta.fai")


def make_bins(cn_by_chrom: dict[str, np.ndarray], depth_per_copy: float = 15.0, gc: float = 0.45) -> pd.DataFrame:
    """100 bp bins with depth proportional to the given copy number, uniform GC, no mask."""
    frames = []
    for chrom, cn in cn_by_chrom.items():
        n = len(cn)
        frames.append(
            pd.DataFrame(
                {
                    "chrom": chrom,
                    "start": np.arange(n, dtype=np.int64) * dc.BIN_SIZE,
                    "end": np.arange(1, n + 1, dtype=np.int64) * dc.BIN_SIZE,
                    "gc": np.float32(gc),
                    "n_frac": np.float32(0),
                    "lowmap": np.float32(0),
                    "segdup": np.float32(0),
                    "depth": (cn * depth_per_copy).astype(np.float32),
                    "depth_uniq": (cn * depth_per_copy).astype(np.float32),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def noisy_cn(n: int, rng: np.random.Generator, level: float = 2.0, sd: float = 0.15) -> np.ndarray:
    return np.clip(level + rng.normal(0, sd, n), 0, None)


class TestPelt:
    def test_single_step_is_found(self):
        x = np.r_[np.zeros(50), np.ones(50) * 3.0]
        ends = dc.pelt_weighted_l2(x, np.ones(100), beta=10.0, min_size=2)
        assert list(ends) == [50, 100]

    def test_flat_data_has_no_changepoint(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 500)
        ends = dc.pelt_weighted_l2(x, np.ones(500), beta=3 * np.log(500), min_size=2)
        assert list(ends) == [500]

    def test_low_weight_hides_a_step(self):
        x = np.r_[np.zeros(50), np.ones(10) * 1.0, np.zeros(50)]
        w = np.ones(110)
        w[50:60] = 1e-3  # noisy units: their step is not worth a changepoint
        assert list(dc.pelt_weighted_l2(x, w, beta=10.0, min_size=2)) == [110]


class TestGivenSex:
    def test_male_and_female_setup(self):
        male, female = dc.sex_chromosome_setup("male"), dc.sex_chromosome_setup("female")
        assert (male.haploid, male.skipped) == ({"chrX", "chrY"}, set())
        assert (female.haploid, female.skipped) == (set(), {"chrY"})
        assert dc.sex_chromosome_setup("male", "X", "Y").haploid == {"X", "Y"}

    def test_invalid_sex(self):
        with pytest.raises(ValueError, match="sex must be"):
            dc.sex_chromosome_setup("M")

    def test_no_call_gives_a_vcf_without_records(self, tmp_path, monkeypatch):
        rng = np.random.default_rng(3)
        bins = make_bins({"chr1": noisy_cn(5000, rng)})  # flat: nothing to call
        monkeypatch.setattr(dc, "load_bins", lambda *a, **k: bins)
        out = str(tmp_path / "calls.vcf.gz")
        fai = tmp_path / "ref.fa.fai"
        fai.write_text("chr1\t500000\t6\t60\t61\n")
        calls = dc.call_depth_cnvs("q0", "q20", "ann", "S", str(fai), out, sex="female")
        assert calls.empty
        with pysam.VariantFile(out) as v:
            assert list(v.header.samples) == ["S"] and not list(v)

    def test_given_sex_overrides_the_depth(self, tmp_path, monkeypatch):
        rng = np.random.default_rng(9)
        bins = make_bins({"chr1": noisy_cn(5000, rng), "chrX": noisy_cn(3000, rng, level=1.0)})  # male depth
        monkeypatch.setattr(dc, "load_bins", lambda *a, **k: bins)
        warnings = []
        monkeypatch.setattr(dc.logger, "warning", warnings.append)
        out = str(tmp_path / "calls.vcf.gz")
        fai = tmp_path / "ref.fa.fai"
        fai.write_text("chr1\t500000\t6\t60\t61\nchrX\t300000\t508480\t60\t61\n")
        calls = dc.call_depth_cnvs("q0", "q20", "ann", "S", str(fai), out, sex="female")
        sex = dc.read_sex_chromosomes(out)
        assert (sex.haploid, sex.skipped, sex.karyotype) == (set(), {"chrY"}, "XX (given)")
        assert any("but the depth gives X " in w for w in warnings)
        assert (calls.chrom == "chrX").any() and set(calls.loc[calls.chrom == "chrX", "direction"]) == {"DEL"}


class TestSexInference:
    def ratios(self, x_level: float, y_level: float | None, chrx="chrX", chry="chrY"):
        rng = np.random.default_rng(7)
        cn = {"chr1": noisy_cn(5000, rng), chrx: noisy_cn(2000, rng, level=x_level)}
        if y_level is not None:
            cn[chry] = noisy_cn(2000, rng, level=y_level, sd=0.05)
        bins = dc.normalize(make_bins(cn), set(), dc.Params())
        return dc.sex_chromosome_ratios(bins, chrx, chry, dc.Params())

    def test_male(self):
        sex = dc.infer_sex_chromosomes(*self.ratios(1.0, 1.0))
        assert (sex.haploid, sex.skipped) == ({"chrX", "chrY"}, set())
        assert sex.karyotype.startswith("XY ")

    def test_female(self):
        sex = dc.infer_sex_chromosomes(*self.ratios(2.0, 0.02))  # chrY: a few mismapped reads
        assert (sex.haploid, sex.skipped) == (set(), {"chrY"})
        assert sex.karyotype.startswith("XX ")

    def test_xxy_and_x0(self):
        xxy = dc.infer_sex_chromosomes(*self.ratios(2.0, 1.0))
        assert (xxy.haploid, xxy.skipped) == ({"chrY"}, set())
        x0 = dc.infer_sex_chromosomes(*self.ratios(1.0, 0.0))
        assert (x0.haploid, x0.skipped) == ({"chrX"}, {"chrY"})

    def test_chrx_only_and_b37_names(self):
        sex = dc.infer_sex_chromosomes(*self.ratios(1.0, None, "X", "Y"), "X", "Y")
        assert (sex.haploid, sex.skipped) == ({"X"}, set())

    def test_no_sex_chromosomes_called(self):
        sex = dc.infer_sex_chromosomes(None, None)
        assert (sex.haploid, sex.skipped) == (set(), set())

    def test_ambiguous_depth_skips_sex_chromosomes(self):
        sex = dc.infer_sex_chromosomes(*self.ratios(1.5, None))
        assert (sex.haploid, sex.skipped) == (set(), {"chrX", "chrY"})
        assert sex.karyotype.startswith("UNDETERMINED")
        y_only = dc.infer_sex_chromosomes(None, 0.5)
        assert y_only.skipped == {"chrX", "chrY"}

    def test_too_few_clean_bins_are_ignored(self):
        rng = np.random.default_rng(8)
        bins = make_bins({"chr1": noisy_cn(5000, rng), "chrX": noisy_cn(2000, rng, level=1.0)})
        bins.loc[bins.chrom == "chrX", "segdup"] = np.float32(1.0)
        bins = dc.normalize(bins, set(), dc.Params())
        assert dc.sex_chromosome_ratios(bins, "chrX", "chrY", dc.Params()) == (None, None)


class TestNormalize:
    def test_copy_number_and_baseline(self):
        rng = np.random.default_rng(1)
        bins = make_bins({"chr1": noisy_cn(5000, rng), "chrX": noisy_cn(2000, rng, level=1.0)})
        out = dc.normalize(bins, {"chrX"}, dc.Params())
        assert out.loc[out.chrom == "chr1", "cn"].median() == pytest.approx(2.0, abs=0.05)
        assert out.loc[out.chrom == "chrX", "cn"].median() == pytest.approx(1.0, abs=0.05)
        assert set(out.loc[out.chrom == "chrX", "baseline"]) == {1.0}

    def test_no_coverage_is_an_error(self):
        bins = make_bins({"chr1": np.zeros(5000)})
        with pytest.raises(ValueError, match="No usable coverage"):
            dc.normalize(bins, set(), dc.Params())

    def test_static_mask_only(self):
        bins = make_bins({"chr1": np.r_[np.full(3000, 2.0), np.zeros(100), np.full(3000, 2.0)]})
        bins.loc[:9, "n_frac"] = np.float32(0.5)
        out = dc.normalize(bins, set(), dc.Params())
        assert not out.loc[:9, "callable"].any()
        assert out.loc[3000:3099, "callable"].all()  # zero coverage is never masked


class TestCalling:
    def call(self, cn: np.ndarray, p: dc.Params | None = None) -> pd.DataFrame:
        p = p or dc.Params()
        bins = dc.normalize(make_bins({"chr1": cn}), set(), p)
        segs = dc.segment(dc.make_units(bins, p), p)
        return dc.merge_calls(segs, bins, p)

    def test_heterozygous_deletion_is_one_call(self):
        rng = np.random.default_rng(2)
        cn = noisy_cn(20000, rng)
        cn[8000:8300] = noisy_cn(300, rng, level=1.0)  # 30 kb deletion
        calls = self.call(cn)
        assert len(calls) == 1
        c = calls.iloc[0]
        assert c.direction == "DEL"
        assert not c["low_conf"]
        assert abs(c.start - 800_000) <= 1000
        assert abs(c.end - 830_000) <= 1000

    def test_masked_gap_is_merged(self):
        rng = np.random.default_rng(3)
        cn = noisy_cn(20000, rng)
        cn[8000:8600] = noisy_cn(600, rng, level=3.0)
        bins = make_bins({"chr1": cn})
        bins.loc[8250:8350, "n_frac"] = np.float32(1.0)  # assembly gap inside the duplication
        p = dc.Params()
        nb = dc.normalize(bins, set(), p)
        calls = dc.merge_calls(dc.segment(dc.make_units(nb, p), p), nb, p)
        assert len(calls) == 1
        assert calls.iloc[0].direction == "DUP"

    def test_normal_sequence_blocks_merge(self):
        rng = np.random.default_rng(4)
        cn = noisy_cn(30000, rng)
        cn[5000:8000] = noisy_cn(3000, rng, level=1.0)  # 300 kb deletion
        cn[8600:11600] = noisy_cn(3000, rng, level=1.0)  # second deletion after 60 kb of normal sequence
        calls = self.call(cn)
        assert len(calls[calls.direction == "DEL"]) == 2

    def test_small_opposite_blip_is_absorbed(self):
        rng = np.random.default_rng(5)
        cn = noisy_cn(30000, rng)
        cn[5000:9000] = noisy_cn(4000, rng, level=1.0)
        cn[7000:7030] = noisy_cn(30, rng, level=3.0)  # 3 kb gain inside a 400 kb deletion
        calls = self.call(cn)
        assert list(calls.direction) == ["DEL"]


class TestOutput:
    def test_vcf_contract(self, tmp_path, fasta_index):
        calls = pd.DataFrame(
            [
                {
                    "chrom": "chr1",
                    "start": 1000,
                    "end": 6000,
                    "direction": "DEL",
                    "low_conf": False,
                    "cn_mean": 1.02,
                    "baseline": 2.0,
                    "z": -12.3,
                    "n_merged": 1,
                    "segdup": 0.0,
                    "uniq_frac": 0.98,
                },
                {
                    "chrom": "chr2",
                    "start": 500,
                    "end": 4500,
                    "direction": "DUP",
                    "low_conf": True,
                    "cn_mean": 2.6,
                    "baseline": 2.0,
                    "z": 7.1,
                    "n_merged": 2,
                    "segdup": 0.8,
                    "uniq_frac": np.nan,
                },
            ]
        )
        out = str(tmp_path / "calls.vcf.gz")
        dc.write_depth_calls(calls, out, "NA24385", fasta_index, dc.SexChromosomes({"chrX"}, {"chrY"}, "X0"))
        sex = dc.read_sex_chromosomes(out)  # how split_read_cnv_candidates gets the setup
        assert (sex.haploid, sex.skipped, sex.karyotype) == ({"chrX"}, {"chrY"}, "X0")
        with pysam.VariantFile(out) as v:
            assert list(v.header.samples) == ["NA24385"]
            for tag in (
                "CN_MEAN",
                "BASE_CN",
                "Z",
                "N_MERGED",
                "SEGDUP",
                "UNIQ_FRAC",
                "LOW_CONF",
                "CIPOS",
                "CIEND",
                "CNV_SOURCE",
            ):
                assert tag in v.header.info
            recs = list(v)
        assert [r.chrom for r in recs] == ["chr1", "chr2"]
        assert recs[0].samples["NA24385"]["GT"] == (0, 1)
        assert recs[1].samples["NA24385"]["GT"] == (None, 1)
        assert recs[0].info["CNV_SOURCE"] == ("depth",)
        assert recs[0].info["BASE_CN"] == 2
        assert recs[0].stop == 6000
        assert "UNIQ_FRAC" not in recs[1].info
        assert [list(r.filter.keys()) for r in recs] == [["PASS"], ["PASS"]]  # FilterVCF decides PASS
        assert "LOW_CONF" not in recs[0].info
        assert recs[1].info["LOW_CONF"]
