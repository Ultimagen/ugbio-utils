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
