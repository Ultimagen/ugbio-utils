"""Unit tests for ugbio_mrd.mrd_detection module."""

import numpy as np
import pandas as pd
import pytest
from ugbio_mrd.mrd_detection import (
    DetectionResult,
    _fit_null_distribution,
    compute_empirical_pvalue,
    compute_personal_lod,
    format_scientific,
    run_detection_analysis,
)


class TestComputeEmpiricalPvalue:
    """Tests for the empirical p-value computation."""

    def test_observed_above_all_null(self):
        """When observed > all null values, p = 1/(S+1)."""
        null_reads = np.array([0, 1, 2, 3, 4])
        p = compute_empirical_pvalue(10, null_reads)
        # (0 + 1) / (5 + 1) = 1/6
        assert p == pytest.approx(1 / 6)

    def test_observed_below_all_null(self):
        """When observed <= all null values, p = 1.0."""
        null_reads = np.array([5, 6, 7, 8, 9])
        p = compute_empirical_pvalue(0, null_reads)
        # (5 + 1) / (5 + 1) = 1.0
        assert p == pytest.approx(1.0)

    def test_observed_equals_some_null(self):
        """When observed equals some null values, they count."""
        null_reads = np.array([0, 1, 2, 3, 3])
        p = compute_empirical_pvalue(3, null_reads)
        # 2 values >= 3, so (2 + 1) / (5 + 1) = 3/6 = 0.5
        assert p == pytest.approx(0.5)

    def test_empty_null(self):
        """With no synthetic controls, p-value is 1.0."""
        p = compute_empirical_pvalue(5, np.array([]))
        assert p == 1.0

    def test_large_null_precise_pvalue(self):
        """With many synthetics, p-value resolution is fine."""
        null_reads = np.zeros(199, dtype=int)
        p = compute_empirical_pvalue(1, null_reads)
        # (0 + 1) / (199 + 1) = 1/200 = 0.005
        assert p == pytest.approx(1 / 200)

    def test_pvalue_conservative_correction(self):
        """p-value is never exactly 0 due to conservative correction."""
        null_reads = np.zeros(1000, dtype=int)
        p = compute_empirical_pvalue(1, null_reads)
        assert p > 0
        assert p == pytest.approx(1 / 1001)


class TestComputePersonalLod:
    """Tests for personal LOD estimation."""

    def test_basic_lod_computation(self):
        """LOD should be computable for reasonable parameters."""
        lod = compute_personal_lod(
            signature_size=1000,
            mean_coverage=40.0,
            denom_ratio=0.5,
            detection_threshold=2,
        )
        assert lod is not None
        assert 1e-7 < lod < 1e-3

    def test_higher_threshold_higher_lod(self):
        """Higher detection threshold should yield higher LOD."""
        lod_low = compute_personal_lod(
            signature_size=1000,
            mean_coverage=40.0,
            denom_ratio=0.5,
            detection_threshold=1,
        )
        lod_high = compute_personal_lod(
            signature_size=1000,
            mean_coverage=40.0,
            denom_ratio=0.5,
            detection_threshold=5,
        )
        assert lod_low < lod_high

    def test_larger_signature_lower_lod(self):
        """Larger signature should yield lower (better) LOD."""
        lod_small = compute_personal_lod(
            signature_size=500,
            mean_coverage=40.0,
            denom_ratio=0.5,
            detection_threshold=2,
        )
        lod_large = compute_personal_lod(
            signature_size=5000,
            mean_coverage=40.0,
            denom_ratio=0.5,
            detection_threshold=2,
        )
        assert lod_small is not None
        assert lod_large is not None
        assert lod_large < lod_small

    def test_zero_signature_returns_none(self):
        """Zero signature size returns None."""
        lod = compute_personal_lod(
            signature_size=0,
            mean_coverage=40.0,
            denom_ratio=0.5,
            detection_threshold=2,
        )
        assert lod is None

    def test_zero_coverage_returns_none(self):
        """Zero coverage returns None."""
        lod = compute_personal_lod(
            signature_size=1000,
            mean_coverage=0.0,
            denom_ratio=0.5,
            detection_threshold=2,
        )
        assert lod is None


class TestRunDetectionAnalysis:
    """Tests for the full detection analysis pipeline."""

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
            denom_ratio=0.5,
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
            denom_ratio=0.5,
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
            denom_ratio=0.5,
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
            denom_ratio=0.5,
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
            denom_ratio=0.5,
        )
        assert result.detected is None
        assert result.call == "Indeterminate"
        assert result.p_value == 1.0


class TestFitNullDistribution:
    """Tests for the automatic distribution selection helper."""

    def test_poisson_for_equidispersed_data(self):
        """Data with variance ≈ mean should get Poisson fit."""
        rng = np.random.default_rng(0)
        null = rng.poisson(3.0, size=200)
        dist, params = _fit_null_distribution(null)
        assert dist == "Poisson"
        assert "lambda" in params
        assert params["lambda"] == pytest.approx(null.mean(), abs=0.5)

    def test_nb_for_overdispersed_data(self):
        """Data with variance >> mean should get Negative Binomial fit."""
        rng = np.random.default_rng(1)
        # Simulate overdispersed counts (NB with r=1, p=0.5 → mu=1, var=2)
        null = rng.negative_binomial(1, 0.5, size=200)
        dist, params = _fit_null_distribution(null)
        assert dist == "NegativeBinomial"
        assert "r" in params and "p" in params and "mu" in params
        assert params["r"] > 0
        assert 0 < params["p"] < 1

    def test_empty_returns_poisson(self):
        """Empty array should return Poisson with lambda=0."""
        dist, params = _fit_null_distribution(np.array([]))
        assert dist == "Poisson"
        assert params["lambda"] == 0.0

    def test_few_samples_returns_poisson(self):
        """Fewer than 3 samples: fall back to Poisson (no variance estimate)."""
        dist, params = _fit_null_distribution(np.array([1, 2]))
        assert dist == "Poisson"

    def test_nb_fitted_pvalue_more_conservative(self):
        """NB p-value should be >= Poisson p-value for the same overdispersed null."""
        from scipy.stats import nbinom, poisson

        rng = np.random.default_rng(2)
        null = rng.negative_binomial(1, 0.5, size=200)
        obs = int(null.max()) + 2  # well above the null

        dist, params = _fit_null_distribution(null)
        assert dist == "NegativeBinomial"

        p_nb = float(nbinom.sf(obs - 1, params["r"], params["p"]))
        p_pois = float(poisson.sf(obs - 1, null.mean()))
        # NB is more conservative (larger p-value) than Poisson for overdispersed data
        assert p_nb >= p_pois


class TestFormatScientific:
    """Tests for scientific notation formatting."""

    def test_zero(self):
        assert format_scientific(0) == "0"

    def test_none(self):
        assert format_scientific(None) == "N/A"

    def test_small_value(self):
        result = format_scientific(3.2e-5)
        assert "10" in result
        assert "5" in result

    def test_power_of_ten(self):
        result = format_scientific(1e-6)
        assert "10" in result
