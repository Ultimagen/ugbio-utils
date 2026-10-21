"""Unit tests for per-class SNVQ recalibration primitives (srsnv_utils + srsnv_training)."""

import subprocess

import numpy as np
import pytest
from ugbio_srsnv.srsnv_training import (
    _count_bases_in_interval_list,
    count_homopolymer_runs_in_interval_list,
    count_str_loci_in_interval_list,
)
from ugbio_srsnv.srsnv_utils import recalibrate_snvq, recalibrate_snvq_per_class, smooth_lut_monotone


def _synth(n=20000, seed=0, tp_shift=8.0):
    """Synthetic MQUAL: TP shifted high, FP low; returns (mqual, label-bool)."""
    rng = np.random.default_rng(seed)
    tp = np.clip(rng.normal(55, 12, n), 0, 100)
    fp = np.clip(rng.normal(20, 12, n) + tp_shift - 8.0, 0, 100)
    mqual = np.concatenate([tp, fp])
    label = np.concatenate([np.ones(n, bool), np.zeros(n, bool)])
    return mqual, label


class TestPrefactorOverride:
    def test_override_matches_manual_formula(self):
        mqual, label = _synth()
        pref = 3e-4
        snvq, x, y = recalibrate_snvq(mqual, label, snvq_prefactor_override=pref, lut_mask=None)
        # recompute y from survival and compare
        tp = mqual[label]
        fp = mqual[~label]
        tpr = np.array([(tp >= m).mean() for m in x])
        fpr = np.array([(fp >= m).mean() for m in x])
        y_exp = -10 * np.log10(np.clip(pref * (fpr / tpr), 1e-12, 1))
        np.testing.assert_allclose(y, y_exp, atol=1e-9)
        assert y.min() >= 0

    def test_override_skips_stats(self):
        # no pos/neg/raw/coverage/n_bases provided -> must not raise
        mqual, label = _synth()
        snvq, x, y = recalibrate_snvq(mqual, label, snvq_prefactor_override=1e-3)
        assert snvq.shape == mqual.shape

    def test_higher_prefactor_lowers_snvq(self):
        mqual, label = _synth()
        _, _, y_lo = recalibrate_snvq(mqual, label, snvq_prefactor_override=1e-4)
        _, _, y_hi = recalibrate_snvq(mqual, label, snvq_prefactor_override=1e-2)
        # a bigger error density -> lower SNVQ at matched MQUAL
        assert np.nanmean(y_hi) < np.nanmean(y_lo)


class TestMonotone:
    def test_monotone_and_cap(self):
        x = np.arange(6, dtype=float)
        y = np.array([10.0, 8.0, 20.0, 19.0, 50.0, 30.0])
        _, ys = smooth_lut_monotone(x, y)
        assert np.all(np.diff(ys) >= 0)
        assert ys.tolist() == [10, 10, 20, 20, 50, 50]
        _, yc = smooth_lut_monotone(x, y, cap=25.0)
        assert yc.max() <= 25.0


class TestPerClass:
    def test_per_class_luts_and_fallback(self):
        mqual_s, lab_s = _synth(n=15000, seed=1)
        mqual_h, lab_h = _synth(n=15000, seed=2)
        mqual = np.concatenate([mqual_s, mqual_h])
        label = np.concatenate([lab_s, lab_h])
        cls = np.array(["snv"] * len(mqual_s) + ["hmer_indel"] * len(mqual_h))
        snvq, luts = recalibrate_snvq_per_class(
            mqual,
            label,
            cls,
            prefactor_by_class={"snv": 1e-4, "hmer_indel": 1e-2},  # hmer penalized harder
        )
        assert set(luts) == {"snv", "hmer_indel"}
        assert not np.isnan(snvq).any()
        # same MQUAL grid, higher-prefactor class scores lower
        m = 55.0
        snv_q = np.interp(m, luts["snv"][0], luts["snv"][1])
        hmer_q = np.interp(m, luts["hmer_indel"][0], luts["hmer_indel"][1])
        assert hmer_q < snv_q

    def test_absent_class_is_nan(self):
        mqual, label = _synth(n=8000)
        cls = np.array(["snv"] * len(mqual))
        snvq, luts = recalibrate_snvq_per_class(
            mqual, label, cls, prefactor_by_class={"snv": 1e-3, "non_hmer_indel": 1e-3}
        )
        assert "non_hmer_indel" not in luts  # no rows -> skipped
        assert not np.isnan(snvq).any()

    def test_monotone_cap_applied_per_class(self):
        mqual, label = _synth(n=8000)
        cls = np.array(["hmer_indel"] * len(mqual))
        snvq, luts = recalibrate_snvq_per_class(
            mqual,
            label,
            cls,
            prefactor_by_class={"hmer_indel": 1e-6},
            smoothing="monotone",
            caps_by_class={"hmer_indel": 40.0},
        )
        assert max(luts["hmer_indel"][1]) <= 40.0 + 1e-9
        assert np.nanmax(snvq) <= 40.0 + 1e-9


class TestIntervalCounters:
    @pytest.fixture
    def tiny_ref(self, tmp_path):
        # chr1: AAAA CG TTT AC-AC-AC ...  -> known homopolymer runs + one STR
        seq = "AAAACGTTTACACACGGG"  # runs: AAAA(4) C G TTT(3) A C A C A C GGG(3); STR ACACAC(unit AC x3)
        fa = tmp_path / "ref.fa"
        fa.write_text(">chr1\n" + seq + "\n")
        subprocess.run(["samtools", "faidx", str(fa)], check=True)
        il = tmp_path / "r.interval_list"
        il.write_text("@HD\tVN:1.0\nchr1\t1\t18\t+\tr1\n")
        return str(fa), str(il), len(seq)

    def test_base_count(self, tiny_ref):
        _fa, il, n = tiny_ref
        assert _count_bases_in_interval_list(il) == n

    def test_hmer_runs(self, tiny_ref):
        fa, il, _n = tiny_ref
        # min_run=2 -> AAAA, TTT, GGG = 3 ; min_run=1 -> every maximal run
        assert count_homopolymer_runs_in_interval_list(il, fa, min_run=2) == 3
        assert count_homopolymer_runs_in_interval_list(il, fa, min_run=4) == 1  # only AAAA
        assert count_homopolymer_runs_in_interval_list(il, fa, min_run=1) >= 3

    def test_str_loci(self, tiny_ref):
        fa, il, _n = tiny_ref
        # ACACAC = unit "AC" x3 -> 1 STR locus (>=2 copies)
        assert count_str_loci_in_interval_list(il, fa, min_unit=2, max_unit=6, min_copies=2) == 1

    def test_sparse_class_falls_back(self):
        mqual, label = _synth(n=8000)
        cls = np.array(["snv"] * len(mqual))
        cls[:30] = "non_hmer_indel"  # 30 TP rows, 0 FP -> below min_lut_rows
        snvq, luts = recalibrate_snvq_per_class(
            mqual, label, cls, prefactor_by_class={"snv": 1e-3, "non_hmer_indel": 1e-3}
        )
        assert "non_hmer_indel" not in luts
        assert np.isnan(snvq[:30]).all()  # caller falls back to the global LUT
        assert not np.isnan(snvq[30:]).any()
