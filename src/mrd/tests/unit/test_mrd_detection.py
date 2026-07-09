"""Unit tests for ugbio_mrd.mrd_detection module."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binom
from ugbio_mrd.mrd_detection import (
    DetectionResult,
    compute_personal_lod,
    format_scientific,
    run_detection_analysis,
)


class TestComputePersonalLod:
    """Tests for personal LOD estimation."""

    def test_basic_lod_computation(self):
        """LOD should be computable for reasonable parameters."""
        lod = compute_personal_lod(
            n=int(1000 * 40.0 * 0.5),
            p_err=1e-6,
        )
        assert lod is not None
        assert 1e-7 < lod < 1e-3

    def test_higher_perr_higher_lod(self):
        """Higher background error rate should yield higher LOD."""
        lod_low = compute_personal_lod(
            n=int(1000 * 40.0 * 0.5),
            p_err=1e-7,
        )
        lod_high = compute_personal_lod(
            n=int(1000 * 40.0 * 0.5),
            p_err=1e-5,
        )
        assert lod_low < lod_high

    def test_larger_signature_lower_lod(self):
        """Larger signature should yield lower (better) LOD."""
        lod_small = compute_personal_lod(
            n=int(500 * 40.0 * 0.5),
            p_err=1e-6,
        )
        lod_large = compute_personal_lod(
            n=int(5000 * 40.0 * 0.5),
            p_err=1e-6,
        )
        assert lod_small is not None
        assert lod_large is not None
        assert lod_large < lod_small

    def test_zero_signature_returns_none(self):
        """Zero n returns None."""
        lod = compute_personal_lod(
            n=0,
            p_err=1e-6,
        )
        assert lod is None

    def test_zero_coverage_returns_none(self):
        """Negative n (guard) returns None."""
        lod = compute_personal_lod(
            n=-1,
            p_err=1e-6,
        )
        assert lod is None

    def test_lod_is_accurate_recall_threshold(self):
        """LOD must be the exact TF where recall crosses 95%, not an abs-residual artefact.

        Regression for the fsolve+abs() bug: brentq gives the true root of the signed
        residual, so recall at LOD should be >= 0.95 and recall just below LOD < 0.95.
        """
        n = int(1000 * 40.0 * 0.5)
        p_err = 1e-6
        fpr = 0.05
        lod = compute_personal_lod(n=n, p_err=p_err)
        assert lod is not None

        # Re-derive n_th to check recall directly
        k_max = int(binom.ppf(0.9999, n, max(p_err, 1e-12))) + 10
        k_range = np.arange(0, k_max + 1)
        sf_values = binom.sf(k_range - 1, n, p_err)
        n_th = int(np.where(sf_values < fpr)[0][0])

        recall_at_lod = binom.sf(n_th - 1, n, p_err + lod)
        recall_below_lod = binom.sf(n_th - 1, n, p_err + lod * 0.5)

        assert recall_at_lod >= 0.95 - 1e-6, f"recall at LOD {recall_at_lod:.6f} < 0.95"
        assert recall_below_lod < 0.95, f"recall below LOD {recall_below_lod:.6f} should be < 0.95"

    @pytest.fixture
    def mock_df_tf_detected(self):
        """Create a df_tf where matched signal clearly exceeds noise.

        Uses 200 synthetic controls (minimum for p < 0.01 per the plan).
        """
        n_synthetics = 200
        # Matched: 10 reads, clearly above noise
        matched_data = {
            "supporting_reads": [10],
            "coverage": [50000],
            "corrected_coverage": [25000],
            "ctdna_vaf": [4e-4],
        }
        # Synthetics: 0-2 reads (noise)
        rng = np.random.default_rng(42)
        syn_reads = rng.integers(0, 3, size=n_synthetics)
        syn_data = {
            "supporting_reads": syn_reads.tolist(),
            "coverage": [50000] * n_synthetics,
            "corrected_coverage": [25000] * n_synthetics,
            "ctdna_vaf": (syn_reads / 25000).tolist(),
        }
        index_matched = pd.MultiIndex.from_tuples(
            [("matched", "patient_sig")],
            names=["signature_type", "signature"],
        )
        index_syn = pd.MultiIndex.from_tuples(
            [("db_control", f"syn{i}") for i in range(n_synthetics)],
            names=["signature_type", "signature"],
        )
        df_matched = pd.DataFrame(matched_data, index=index_matched)
        df_syn = pd.DataFrame(syn_data, index=index_syn)
        return pd.concat([df_matched, df_syn])

    @pytest.fixture
    def mock_df_tf_not_detected(self):
        """Create a df_tf where matched signal is within noise."""
        n_synthetics = 200
        # Matched: 1 read, within noise range
        matched_data = {
            "supporting_reads": [1],
            "coverage": [50000],
            "corrected_coverage": [25000],
            "ctdna_vaf": [4e-5],
        }
        # Synthetics: 0-2 reads (noise), some will be >= 1
        rng = np.random.default_rng(42)
        syn_reads = rng.integers(0, 3, size=n_synthetics)
        syn_data = {
            "supporting_reads": syn_reads.tolist(),
            "coverage": [50000] * n_synthetics,
            "corrected_coverage": [25000] * n_synthetics,
            "ctdna_vaf": (syn_reads / 25000).tolist(),
        }
        index_matched = pd.MultiIndex.from_tuples(
            [("matched", "patient_sig")],
            names=["signature_type", "signature"],
        )
        index_syn = pd.MultiIndex.from_tuples(
            [("db_control", f"syn{i}") for i in range(n_synthetics)],
            names=["signature_type", "signature"],
        )
        df_matched = pd.DataFrame(matched_data, index=index_matched)
        df_syn = pd.DataFrame(syn_data, index=index_syn)
        return pd.concat([df_matched, df_syn])

    @pytest.fixture
    def mock_df_signatures_filt(self):
        """Mock filtered signature dataframe."""
        n_loci = 1000
        index = pd.MultiIndex.from_arrays(
            [
                [f"chr{i % 22 + 1}" for i in range(n_loci)],
                list(range(1000, 1000 + n_loci)),
            ],
            names=["chrom", "pos"],
        )
        return pd.DataFrame(
            {
                "signature_type": ["matched"] * n_loci,
                "signature": ["patient_sig"] * n_loci,
                "coverage": np.random.default_rng(42).integers(20, 60, n_loci),
            },
            index=index,
        )

    def test_detected_result(self, mock_df_tf_detected, mock_df_signatures_filt):
        """Clear signal should yield MRD Detected."""
        result = run_detection_analysis(
            df_tf=mock_df_tf_detected,
            df_signatures_filt=mock_df_signatures_filt,
        )
        assert isinstance(result, DetectionResult)
        assert result.detected is True
        assert result.call == "MRD Detected"
        assert result.p_value < 0.01
        assert result.matched_supporting_reads == 10

    def test_not_detected_result(self, mock_df_tf_not_detected, mock_df_signatures_filt):
        """Signal within noise should yield MRD Not Detected."""
        result = run_detection_analysis(
            df_tf=mock_df_tf_not_detected,
            df_signatures_filt=mock_df_signatures_filt,
        )
        assert isinstance(result, DetectionResult)
        assert result.detected is False
        assert result.call == "MRD Not Detected"
        assert result.p_value > 0.01

    def test_personal_lod_is_computed(self, mock_df_tf_detected, mock_df_signatures_filt):
        """Personal LOD should be computed when data is available."""
        result = run_detection_analysis(
            df_tf=mock_df_tf_detected,
            df_signatures_filt=mock_df_signatures_filt,
        )
        assert result.personal_lod is not None
        assert 1e-7 < result.personal_lod < 1e-3

    def test_no_matched_signature(self, mock_df_signatures_filt):
        """Missing matched signature yields Indeterminate."""
        data = {
            "supporting_reads": [0, 1],
            "coverage": [50000, 50000],
            "corrected_coverage": [25000, 25000],
            "ctdna_vaf": [0, 4e-5],
        }
        index = pd.MultiIndex.from_tuples(
            [("db_control", "syn0"), ("db_control", "syn1")],
            names=["signature_type", "signature"],
        )
        df_tf = pd.DataFrame(data, index=index)
        result = run_detection_analysis(
            df_tf=df_tf,
            df_signatures_filt=mock_df_signatures_filt,
        )
        assert result.detected is None
        assert result.call == "Indeterminate"

    def test_no_synthetic_controls(self, mock_df_signatures_filt):
        """No db_control entries yields Indeterminate."""
        data = {
            "supporting_reads": [5],
            "coverage": [50000],
            "corrected_coverage": [25000],
            "ctdna_vaf": [2e-4],
        }
        index = pd.MultiIndex.from_tuples(
            [("matched", "patient_sig")],
            names=["signature_type", "signature"],
        )
        df_tf = pd.DataFrame(data, index=index)
        result = run_detection_analysis(
            df_tf=df_tf,
            df_signatures_filt=mock_df_signatures_filt,
        )
        assert result.detected is None
        assert result.call == "Indeterminate"
        assert result.p_value == 1.0

    def test_zero_corrected_coverage_is_indeterminate(self, mock_df_signatures_filt):
        """Controls present but corrected_coverage==0 must yield Indeterminate, not a false detection.

        Regression: previously p_err was forced to 0.0 in this branch, causing
        binom.sf(obs-1, n, 0) == 0 for any positive obs -> spurious 'MRD Detected'.
        """
        matched_data = {
            "supporting_reads": [10],
            "coverage": [50000],
            "corrected_coverage": [25000],
            "ctdna_vaf": [4e-4],
        }
        # Synthetic controls present but with zero corrected_coverage
        syn_data = {
            "supporting_reads": [0, 0, 1],
            "coverage": [0, 0, 0],
            "corrected_coverage": [0, 0, 0],  # <- invalid depth
            "ctdna_vaf": [0.0, 0.0, 0.0],
        }
        index_matched = pd.MultiIndex.from_tuples([("matched", "patient_sig")], names=["signature_type", "signature"])
        index_syn = pd.MultiIndex.from_tuples(
            [("db_control", f"syn{i}") for i in range(3)], names=["signature_type", "signature"]
        )
        df_tf = pd.concat(
            [
                pd.DataFrame(matched_data, index=index_matched),
                pd.DataFrame(syn_data, index=index_syn),
            ]
        )
        result = run_detection_analysis(
            df_tf=df_tf,
            df_signatures_filt=mock_df_signatures_filt,
        )
        assert result.call == "Indeterminate", f"Expected Indeterminate when db_control coverage=0, got {result.call}"


class TestMultiReadSupportQcCheck:
    """Tests for the per-locus outlier detection QC checks (matched + controls)."""

    @pytest.fixture
    def base_df_tf(self):
        """Minimal df_tf with matched signal and enough synthetic controls to pass other QC checks."""
        n_synthetics = 25
        rng = np.random.default_rng(0)
        syn_reads = rng.integers(0, 2, size=n_synthetics)
        index_matched = pd.MultiIndex.from_tuples([("matched", "patient_sig")], names=["signature_type", "signature"])
        index_syn = pd.MultiIndex.from_tuples(
            [("db_control", f"syn{i}") for i in range(n_synthetics)],
            names=["signature_type", "signature"],
        )
        df_matched = pd.DataFrame(
            {"supporting_reads": [5], "coverage": [50000], "corrected_coverage": [25000], "ctdna_vaf": [2e-4]},
            index=index_matched,
        )
        df_syn = pd.DataFrame(
            {
                "supporting_reads": syn_reads.tolist(),
                "coverage": [50000] * n_synthetics,
                "corrected_coverage": [25000] * n_synthetics,
                "ctdna_vaf": (syn_reads / 25000).tolist(),
            },
            index=index_syn,
        )
        return pd.concat([df_matched, df_syn])

    @pytest.fixture
    def base_df_signatures_filt(self):
        """Mock signature with 1000 loci and 30× mean coverage."""
        n_loci = 1000
        index = pd.MultiIndex.from_arrays(
            [
                [f"chr{i % 22 + 1}" for i in range(n_loci)],
                list(range(1000, 1000 + n_loci)),
            ],
            names=["chrom", "pos"],
        )
        return pd.DataFrame(
            {
                "signature_type": ["matched"] * n_loci,
                "signature": ["patient_sig"] * n_loci,
                "coverage": [30] * n_loci,
            },
            index=index,
        )

    def _make_per_locus_df(self, reads_list: list[int], sig_type: str = "matched", sig_name: str = "sig0"):
        """Build a df_supporting_reads_per_locus fragment with a (chrom, pos) MultiIndex."""
        n = len(reads_list)
        index = pd.MultiIndex.from_arrays(
            [[f"chr{i % 22 + 1}" for i in range(n)], list(range(1000, 1000 + n))],
            names=["chrom", "pos"],
        )
        return pd.DataFrame(
            {
                "signature": [sig_name] * n,
                "signature_type": [sig_type] * n,
                "supporting_reads": reads_list,
            },
            index=index,
        )

    def _get_matched_qc(self, result):
        """Extract the matched multi-read support QC check."""
        matches = [c for c in result.qc_checks if c.label == "Expected multi-read support distribution (matched)"]
        assert len(matches) == 1, f"Expected 1 matched multi-read QC check, got {len(matches)}"
        return matches[0]

    def _get_control_qc(self, result, ctrl_label: str):
        """Extract a control multi-read support QC check by control label."""
        matches = [c for c in result.qc_checks if ctrl_label in c.label]
        assert len(matches) == 1, f"Expected 1 '{ctrl_label}' QC check, got {len(matches)}"
        return matches[0]

    # ── matched signature checks ──────────────────────────────────────────────

    def test_no_outliers_passes(self, base_df_tf, base_df_signatures_filt):
        """All loci with 1 read at low TF should produce no Bonferroni outliers."""
        per_locus = self._make_per_locus_df([1] * 1000)
        result = run_detection_analysis(
            df_tf=base_df_tf,
            df_signatures_filt=base_df_signatures_filt,
            df_supporting_reads_per_locus=per_locus,
        )
        qc = self._get_matched_qc(result)
        assert qc.passed is True
        assert "0 outlier loci" in qc.value_str

    def test_single_read_loci_not_flagged_at_very_low_tf(self, base_df_signatures_filt):
        """Single-read loci must never be counted as QC outliers, even at near-zero TF.

        When estimated TF is essentially zero, λ → 0 and poisson.sf(0, λ) ≈ λ is tiny,
        so the Bonferroni-corrected p-value for a single read can fall below the 1%
        threshold.  Without the ≥2-read guard the QC check would falsely flag those loci
        even though apply_multi_read_locus_filter deliberately never removes them.
        """
        n_synthetics = 30
        rng = np.random.default_rng(42)
        syn_reads = rng.integers(0, 2, size=n_synthetics)
        index_matched = pd.MultiIndex.from_tuples([("matched", "patient_sig")], names=["signature_type", "signature"])
        index_syn = pd.MultiIndex.from_tuples(
            [("db_control", f"syn{i}") for i in range(n_synthetics)],
            names=["signature_type", "signature"],
        )
        df_matched = pd.DataFrame(
            {
                "supporting_reads": [1],
                "coverage": [50000],
                "corrected_coverage": [25000],
                "ctdna_vaf": [1e-8],  # near-zero TF → λ = 1e-8 * 30 ≈ 0
            },
            index=index_matched,
        )
        df_syn = pd.DataFrame(
            {
                "supporting_reads": syn_reads.tolist(),
                "coverage": [50000] * n_synthetics,
                "corrected_coverage": [25000] * n_synthetics,
                "ctdna_vaf": (syn_reads / 25000).tolist(),
            },
            index=index_syn,
        )
        df_tf_near_zero = pd.concat([df_matched, df_syn])
        per_locus = self._make_per_locus_df([1] * 1000)  # 1000 single-read loci
        result = run_detection_analysis(
            df_tf=df_tf_near_zero,
            df_signatures_filt=base_df_signatures_filt,
            df_supporting_reads_per_locus=per_locus,
        )
        qc = self._get_matched_qc(result)
        assert (
            qc.passed is True
        ), f"Single-read loci must not be flagged as outliers at near-zero TF; got: {qc.value_str}"
        assert "0 outlier loci" in qc.value_str

    def test_germline_outlier_fails(self, base_df_tf, base_df_signatures_filt):
        """A single locus with many reads (germline-like) should be flagged as outlier."""
        reads = [0] * 999 + [15]
        per_locus = self._make_per_locus_df(reads)
        result = run_detection_analysis(
            df_tf=base_df_tf,
            df_signatures_filt=base_df_signatures_filt,
            df_supporting_reads_per_locus=per_locus,
        )
        qc = self._get_matched_qc(result)
        assert qc.passed is False
        assert "1 outlier locus" in qc.value_str

    def test_multiple_outliers_fail(self, base_df_tf, base_df_signatures_filt):
        """Multiple high-support loci should each be counted as outliers."""
        reads = [0] * 995 + [12, 14, 11, 13, 10]
        per_locus = self._make_per_locus_df(reads)
        result = run_detection_analysis(
            df_tf=base_df_tf,
            df_signatures_filt=base_df_signatures_filt,
            df_supporting_reads_per_locus=per_locus,
        )
        qc = self._get_matched_qc(result)
        assert qc.passed is False
        assert "outlier loci" in qc.value_str

    def test_matched_label(self, base_df_tf, base_df_signatures_filt):
        """Matched QC check label must include '(matched)'."""
        per_locus = self._make_per_locus_df([1] * 1000)
        result = run_detection_analysis(
            df_tf=base_df_tf,
            df_signatures_filt=base_df_signatures_filt,
            df_supporting_reads_per_locus=per_locus,
        )
        qc = self._get_matched_qc(result)
        assert qc.label == "Expected multi-read support distribution (matched)"

    def test_no_per_locus_df_skips_all_checks(self, base_df_tf, base_df_signatures_filt):
        """When df_supporting_reads_per_locus is None all multi-read QC checks are absent."""
        result = run_detection_analysis(
            df_tf=base_df_tf,
            df_signatures_filt=base_df_signatures_filt,
            df_supporting_reads_per_locus=None,
        )
        matches = [c for c in result.qc_checks if "multi-read" in c.label.lower()]
        assert len(matches) == 0

    # ── control checks ────────────────────────────────────────────────────────

    def test_db_control_no_outliers_passes(self, base_df_tf, base_df_signatures_filt):
        """db_control loci with 0 reads should pass the synthetic-controls check."""
        matched_loci = self._make_per_locus_df([1] * 1000, sig_type="matched")
        ctrl_loci = self._make_per_locus_df([0] * 1000, sig_type="db_control", sig_name="syn0")
        per_locus = pd.concat([matched_loci, ctrl_loci])
        result = run_detection_analysis(
            df_tf=base_df_tf,
            df_signatures_filt=base_df_signatures_filt,
            df_supporting_reads_per_locus=per_locus,
        )
        qc = self._get_control_qc(result, "synthetic controls")
        assert qc.passed is True

    def test_db_control_outlier_fails(self, base_df_tf, base_df_signatures_filt):
        """A single db_control locus with many reads should fail the synthetic-controls check."""
        matched_loci = self._make_per_locus_df([0] * 1000, sig_type="matched")
        ctrl_reads = [0] * 999 + [12]
        ctrl_loci = self._make_per_locus_df(ctrl_reads, sig_type="db_control", sig_name="syn0")
        per_locus = pd.concat([matched_loci, ctrl_loci])
        result = run_detection_analysis(
            df_tf=base_df_tf,
            df_signatures_filt=base_df_signatures_filt,
            df_supporting_reads_per_locus=per_locus,
        )
        qc = self._get_control_qc(result, "synthetic controls")
        assert qc.passed is False

    def test_cohort_control_outlier_fails(self, base_df_tf, base_df_signatures_filt):
        """A cohort control locus with many reads should fail the cohort-controls check."""
        matched_loci = self._make_per_locus_df([0] * 1000, sig_type="matched")
        cohort_reads = [0] * 998 + [10, 11]
        cohort_loci = self._make_per_locus_df(cohort_reads, sig_type="control", sig_name="cohort0")
        per_locus = pd.concat([matched_loci, cohort_loci])
        result = run_detection_analysis(
            df_tf=base_df_tf,
            df_signatures_filt=base_df_signatures_filt,
            df_supporting_reads_per_locus=per_locus,
        )
        qc = self._get_control_qc(result, "cohort controls")
        assert qc.passed is False

    def test_cohort_control_no_outliers_passes(self, base_df_tf, base_df_signatures_filt):
        """Cohort control loci with 0 reads should pass the cohort-controls check."""
        matched_loci = self._make_per_locus_df([1] * 1000, sig_type="matched")
        cohort_loci = self._make_per_locus_df([0] * 1000, sig_type="control", sig_name="cohort0")
        per_locus = pd.concat([matched_loci, cohort_loci])
        result = run_detection_analysis(
            df_tf=base_df_tf,
            df_signatures_filt=base_df_signatures_filt,
            df_supporting_reads_per_locus=per_locus,
        )
        qc = self._get_control_qc(result, "cohort controls")
        assert qc.passed is True
        assert "0 outlier loci" in qc.value_str

    def test_cohort_control_absent_skips_check(self, base_df_tf, base_df_signatures_filt):
        """When no cohort control loci are present the cohort QC check is absent."""
        per_locus = self._make_per_locus_df([1] * 1000, sig_type="matched")
        result = run_detection_analysis(
            df_tf=base_df_tf,
            df_signatures_filt=base_df_signatures_filt,
            df_supporting_reads_per_locus=per_locus,
        )
        matches = [c for c in result.qc_checks if "cohort controls" in c.label]
        assert len(matches) == 0

    def test_control_check_uses_max_across_signatures(self, base_df_tf, base_df_signatures_filt):
        """For db_control with multiple signatures, the worst locus (max reads) drives the check."""
        matched_loci = self._make_per_locus_df([0] * 1000, sig_type="matched")
        # Same 1000 loci, two synthetic signatures: one clean, one with a single high-count locus
        ctrl_clean = self._make_per_locus_df([0] * 1000, sig_type="db_control", sig_name="syn0")
        ctrl_hot = self._make_per_locus_df([0] * 999 + [15], sig_type="db_control", sig_name="syn1")
        per_locus = pd.concat([matched_loci, ctrl_clean, ctrl_hot])
        result = run_detection_analysis(
            df_tf=base_df_tf,
            df_signatures_filt=base_df_signatures_filt,
            df_supporting_reads_per_locus=per_locus,
        )
        qc = self._get_control_qc(result, "synthetic controls")
        assert qc.passed is False


class TestFormatScientific:
    """Tests for scientific notation formatting."""

    def test_zero(self):
        assert format_scientific(0) == "0"

    def test_none(self):
        assert format_scientific(None) == "N/A"

    def test_small_value(self):
        result = format_scientific(3.2e-5)
        assert "10" in result
        assert "\u207b" in result  # superscript minus
        assert "3.2" in result

    def test_power_of_ten(self):
        result = format_scientific(1e-6)
        assert "10" in result
