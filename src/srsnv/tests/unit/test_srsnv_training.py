import argparse
import gzip
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import polars as pl
import pytest
import xgboost as xgb
from ugbio_featuremap.featuremap_utils import FeatureMapFields
from ugbio_srsnv.split_manifest import (
    SPLIT_MODE_SINGLE_MODEL_CHROM_VAL,
    SPLIT_MODE_SINGLE_MODEL_READ_HASH,
)
from ugbio_srsnv.srsnv_training import (
    CHROM,
    FILTERS_FULL_OUTPUT,
    FILTERS_RANDOM_SAMPLE,
    FOLD_COL,
    LABEL_COL,
    MQUAL,
    POS,
    PROB_ORIG,
    REF,
    SNVQ,
    X_ALT,
    X_HMER_ALT,
    X_HMER_REF,
    SRSNVTrainer,
    _build_new_split_manifest,
    _configure_xgb_device,
    _count_bases_in_interval_list,
    _extract_stats_from_unified,
    _last_non_downsample_funnel,
    _parse_holdout_chromosomes,
    _parse_model_params,
    _parse_user_metadata,
    _probability_recalibration,
    _validate_quality_region_filters,
    partition_into_folds,
)


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, {}),
        (
            "eta=0.1:max_depth=8:n_estimators=200",
            {"eta": 0.1, "max_depth": 8, "n_estimators": 200},
        ),
        ("verbosity=debug:subsample=0.9", {"verbosity": "debug", "subsample": 0.9}),
    ],
)
def test_parse_model_params_inline(raw: str | None, expected: dict[str, Any]) -> None:
    assert _parse_model_params(raw) == expected


def test_parse_model_params_json(tmp_path: Path) -> None:  # noqa: D103
    json_path = tmp_path / "xgb_params.json"
    payload = {"eta": 0.05, "max_depth": 6, "enable_categorical": True}
    json_path.write_text(json.dumps(payload))

    parsed = _parse_model_params(str(json_path))
    assert parsed == payload


def test_parse_model_params_invalid() -> None:  # noqa: D103
    with pytest.raises(ValueError):
        _parse_model_params("eta=0.1:max_depth")  # uneven tokens
    with pytest.raises(ValueError):
        _parse_model_params("eta")  # missing '='


def test_extract_stats_from_unified_new_format(resources_dir):
    """Test _extract_stats_from_unified with new format (filtering_stats sections)."""
    stats_file = resources_dir / "402572-CL10377.model_filters_status.funnel.edited.json"
    pos_stats, neg_stats = _extract_stats_from_unified(stats_file)

    # Verify structure
    assert "filters" in pos_stats
    assert "filters" in neg_stats

    # Verify some basic content
    assert isinstance(pos_stats["filters"], list)
    assert isinstance(neg_stats["filters"], list)
    assert len(pos_stats["filters"]) > 0
    assert len(neg_stats["filters"]) > 0

    # Verify that raw filter is present
    raw_filter_pos = next((f for f in pos_stats["filters"] if f["name"] == "raw"), None)
    raw_filter_neg = next((f for f in neg_stats["filters"] if f["name"] == "raw"), None)
    assert raw_filter_pos is not None
    assert raw_filter_neg is not None
    assert raw_filter_pos["type"] == "raw"
    assert raw_filter_neg["type"] == "raw"

    # Verify combinations data is preserved
    assert "combinations" in pos_stats, "Positive stats should contain 'combinations'"
    assert "combinations" in neg_stats, "Negative stats should contain 'combinations'"
    assert isinstance(pos_stats["combinations"], dict)
    assert isinstance(neg_stats["combinations"], dict)
    assert len(pos_stats["combinations"]) > 0
    assert len(neg_stats["combinations"]) > 0

    # Verify combinations_total is preserved
    assert "combinations_total" in pos_stats, "Positive stats should contain 'combinations_total'"
    assert "combinations_total" in neg_stats, "Negative stats should contain 'combinations_total'"
    assert pos_stats["combinations_total"] == 2725
    assert neg_stats["combinations_total"] == 62256269


def test_extract_stats_from_unified_missing_section():
    """Test _extract_stats_from_unified with missing required sections."""

    # Test missing filters_random_sample
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({FILTERS_FULL_OUTPUT: {"filters": {}}}, f)
        temp_path = f.name

    with pytest.raises(ValueError, match=f"missing {FILTERS_RANDOM_SAMPLE} section"):
        _extract_stats_from_unified(temp_path)

    # Clean up
    Path(temp_path).unlink()

    # Test missing filters_full_output
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({FILTERS_RANDOM_SAMPLE: {"filters": {}}}, f)
        temp_path = f.name

    with pytest.raises(ValueError, match=f"missing {FILTERS_FULL_OUTPUT} section"):
        _extract_stats_from_unified(temp_path)

    # Clean up
    Path(temp_path).unlink()


def test_downsample_segments_added_to_metadata(tmp_path: Path, resources_dir: Path) -> None:
    """Test that downsample segments are added to positive and negative stats in metadata."""
    # Setup paths
    pos_file = resources_dir / "402572-CL10377.random_sample.featuremap.filtered.parquet"
    neg_file = resources_dir / "402572-CL10377.raw.featuremap.filtered.parquet"
    stats_file = resources_dir / "402572-CL10377.model_filters_status.funnel.edited.json"
    bed_file = resources_dir / "wgs_calling_regions.without_encode_blacklist.hg38.chr1_22.interval_list"

    # Create args for trainer
    args = argparse.Namespace(
        positive=str(pos_file),
        negative=str(neg_file),
        stats_file=str(stats_file),
        mean_coverage=30.0,
        training_regions=str(bed_file),
        k_folds=1,
        model_params="n_estimators=2:max_depth=2:enable_categorical=true",
        features="REF:ALT:X_HMER_REF:X_HMER_ALT",
        output=str(tmp_path),
        basename="test_downsample.",
        random_seed=0,
        verbose=False,
        max_qual=100.0,
        quality_lut_size=100,
        metadata=None,
        use_gpu=False,
        use_float32=False,
        use_kde_smoothing=False,
    )

    # Initialize and run trainer
    trainer = SRSNVTrainer(args)
    trainer.train()
    trainer.save()

    # Load metadata
    metadata_path = tmp_path / "test_downsample.srsnv_metadata.json"
    assert metadata_path.is_file(), "metadata file not created"

    with metadata_path.open() as f:
        metadata = json.load(f)

    # Check that filtering_stats has positive and negative
    assert "filtering_stats" in metadata
    assert "positive" in metadata["filtering_stats"]
    assert "negative" in metadata["filtering_stats"]

    pos_stats = metadata["filtering_stats"]["positive"]
    neg_stats = metadata["filtering_stats"]["negative"]

    # Check that both have filters list
    assert "filters" in pos_stats
    assert "filters" in neg_stats

    # Find all downsample segments
    pos_downsample = [f for f in pos_stats["filters"] if f.get("type") == "downsample"]
    neg_downsample = [f for f in neg_stats["filters"] if f.get("type") == "downsample"]

    # Verify exactly one downsample segment exists in each
    assert (
        len(pos_downsample) == 1
    ), f"Expected exactly one downsample segment in positive stats, found {len(pos_downsample)}"
    assert (
        len(neg_downsample) == 1
    ), f"Expected exactly one downsample segment in negative stats, found {len(neg_downsample)}"

    # Get the downsample segments
    pos_ds = pos_downsample[0]
    neg_ds = neg_downsample[0]

    # Verify the downsample segment is the last filter
    assert pos_stats["filters"][-1] == pos_ds, "Downsample segment should be the last filter in positive stats"
    assert neg_stats["filters"][-1] == neg_ds, "Downsample segment should be the last filter in negative stats"
    assert pos_ds["name"] == "downsample"
    assert pos_ds["type"] == "downsample"
    assert pos_ds["method"] == "random"
    assert pos_ds["seed"] == 0
    assert "funnel" in pos_ds
    assert isinstance(pos_ds["funnel"], int)
    assert pos_ds["funnel"] > 0
    # Verify pass field is present and matches funnel for positive
    assert "pass" in pos_ds, "pass field should be present in positive downsample segment"
    assert pos_ds["pass"] == pos_ds["funnel"], "pass should equal funnel for positive downsample"

    # Verify downsample segment structure for negative
    assert neg_ds["name"] == "downsample"
    assert neg_ds["type"] == "downsample"
    assert neg_ds["method"] == "random"
    assert neg_ds["seed"] == 0
    assert "funnel" in neg_ds
    assert isinstance(neg_ds["funnel"], int)
    assert neg_ds["funnel"] > 0
    # Verify pass field is present and matches funnel for negative
    assert "pass" in neg_ds, "pass field should be present in negative downsample segment"
    assert neg_ds["pass"] == neg_ds["funnel"], "pass should equal funnel for negative downsample"

    # Verify that row counts match the actual data loaded from parquet files
    # The trainer's data_frame should have total rows equal to pos + neg
    expected_total = pos_ds["funnel"] + neg_ds["funnel"]
    assert expected_total == trainer.data_frame.height, (
        f"Downsample row counts ({pos_ds['funnel']} + {neg_ds['funnel']} = {expected_total}) "
        f"don't match total data frame height ({trainer.data_frame.height})"
    )
    assert (
        neg_ds["funnel"] == trainer.n_neg
    ), f"Negative downsample rows ({neg_ds['funnel']}) don't match trainer.n_neg ({trainer.n_neg})"

    # Verify combinations data is preserved in metadata output
    assert "combinations" in pos_stats, "Positive stats in metadata should contain 'combinations'"
    assert "combinations" in neg_stats, "Negative stats in metadata should contain 'combinations'"
    assert isinstance(pos_stats["combinations"], dict)
    assert isinstance(neg_stats["combinations"], dict)
    assert "combinations_total" in pos_stats, "Positive stats in metadata should contain 'combinations_total'"
    assert "combinations_total" in neg_stats, "Negative stats in metadata should contain 'combinations_total'"


# ──────────────────────── _parse_holdout_chromosomes ──────────────────────


class TestParseHoldoutChromosomes:
    """Tests for _parse_holdout_chromosomes."""

    def test_none_returns_none(self):
        assert _parse_holdout_chromosomes(None) is None

    def test_empty_string_returns_none(self):
        assert _parse_holdout_chromosomes("") is None

    def test_whitespace_only_returns_none(self):
        assert _parse_holdout_chromosomes("  ,  , ") is None

    def test_single_chromosome(self):
        result = _parse_holdout_chromosomes("chr21")
        assert result == ["chr21"]

    def test_multiple_chromosomes(self):
        result = _parse_holdout_chromosomes("chr21,chr22,chrX")
        assert result == ["chr21", "chr22", "chrX"]

    def test_strips_whitespace(self):
        result = _parse_holdout_chromosomes(" chr21 , chr22 ")
        assert result == ["chr21", "chr22"]

    def test_deduplicates_preserving_order(self):
        result = _parse_holdout_chromosomes("chr21,chr22,chr21,chrX,chr22")
        assert result == ["chr21", "chr22", "chrX"]

    def test_single_comma(self):
        result = _parse_holdout_chromosomes(",")
        assert result is None


# ──────────────────────── _validate_quality_region_filters ──────────────


class TestValidateQualityRegionFilters:
    """Tests for _validate_quality_region_filters."""

    def test_matching_filters_pass(self):
        """Identical quality/region filters should not raise."""
        pos_stats = {
            "filters": [
                {"name": "raw", "type": "raw", "funnel": 1000, "pass": 1000},
                {"name": "qual_filter", "type": "quality", "threshold": 30},
                {"name": "region_filter", "type": "region", "bed_file": "regions.bed"},
                {"name": "downsample", "type": "downsample", "funnel": 100, "pass": 100},
            ]
        }
        neg_stats = {
            "filters": [
                {"name": "raw", "type": "raw", "funnel": 5000, "pass": 5000},
                {"name": "qual_filter", "type": "quality", "threshold": 30},
                {"name": "region_filter", "type": "region", "bed_file": "regions.bed"},
                {"name": "downsample", "type": "downsample", "funnel": 500, "pass": 500},
            ]
        }
        # Should not raise
        _validate_quality_region_filters(pos_stats, neg_stats)

    def test_mismatched_quality_filter_raises(self):
        """Different quality filters should raise ValueError."""
        pos_stats = {
            "filters": [
                {"name": "qual_filter", "type": "quality", "threshold": 30},
            ]
        }
        neg_stats = {
            "filters": [
                {"name": "qual_filter", "type": "quality", "threshold": 20},
            ]
        }
        with pytest.raises(ValueError, match="Mismatch between quality/region filters"):
            _validate_quality_region_filters(pos_stats, neg_stats)

    def test_mismatched_region_filter_raises(self):
        """Different region filters should raise ValueError."""
        pos_stats = {
            "filters": [
                {"name": "region_filter", "type": "region", "bed_file": "regions_a.bed"},
            ]
        }
        neg_stats = {
            "filters": [
                {"name": "region_filter", "type": "region", "bed_file": "regions_b.bed"},
            ]
        }
        with pytest.raises(ValueError, match="Mismatch between quality/region filters"):
            _validate_quality_region_filters(pos_stats, neg_stats)

    def test_no_quality_region_filters_is_ok(self):
        """Both sides having no quality/region filters should pass."""
        pos_stats = {
            "filters": [
                {"name": "raw", "type": "raw", "funnel": 1000, "pass": 1000},
                {"name": "downsample", "type": "downsample", "funnel": 100, "pass": 100},
            ]
        }
        neg_stats = {
            "filters": [
                {"name": "raw", "type": "raw", "funnel": 5000, "pass": 5000},
                {"name": "downsample", "type": "downsample", "funnel": 500, "pass": 500},
            ]
        }
        # Should not raise
        _validate_quality_region_filters(pos_stats, neg_stats)

    def test_ignores_funnel_pass_in_comparison(self):
        """funnel and pass fields should be excluded from comparison."""
        pos_stats = {
            "filters": [
                {"name": "qual_filter", "type": "quality", "threshold": 30, "funnel": 100, "pass": 90},
            ]
        }
        neg_stats = {
            "filters": [
                {"name": "qual_filter", "type": "quality", "threshold": 30, "funnel": 5000, "pass": 4500},
            ]
        }
        # Should not raise (funnel and pass are excluded from comparison)
        _validate_quality_region_filters(pos_stats, neg_stats)


# ──────────────────────── _last_non_downsample_funnel ──────────────────────


class TestLastNonDownsampleFunnel:
    """Tests for _last_non_downsample_funnel."""

    def test_returns_last_non_downsample_funnel(self):
        """Should return the funnel value of the last non-downsample filter."""
        stats = {
            "filters": [
                {"name": "raw", "type": "raw", "funnel": 10000},
                {"name": "quality", "type": "quality", "funnel": 5000},
                {"name": "region", "type": "region", "funnel": 3000},
                {"name": "downsample", "type": "downsample", "funnel": 1000},
            ]
        }
        result = _last_non_downsample_funnel(stats)
        assert result == 3000

    def test_no_downsample_returns_last_entry(self):
        """When there are no downsample entries, return the last filter's funnel."""
        stats = {
            "filters": [
                {"name": "raw", "type": "raw", "funnel": 10000},
                {"name": "quality", "type": "quality", "funnel": 7000},
            ]
        }
        result = _last_non_downsample_funnel(stats)
        assert result == 7000

    def test_multiple_downsample_at_end(self):
        """Should skip all downsample entries at the end."""
        stats = {
            "filters": [
                {"name": "raw", "type": "raw", "funnel": 10000},
                {"name": "region", "type": "region", "funnel": 4000},
                {"name": "downsample1", "type": "downsample", "funnel": 2000},
                {"name": "downsample2", "type": "downsample", "funnel": 500},
            ]
        }
        result = _last_non_downsample_funnel(stats)
        assert result == 4000

    def test_all_downsample_raises(self):
        """When all filters are downsample, should raise ValueError."""
        stats = {
            "filters": [
                {"name": "downsample1", "type": "downsample", "funnel": 1000},
                {"name": "downsample2", "type": "downsample", "funnel": 500},
            ]
        }
        with pytest.raises(ValueError, match="no non-downsample filter entry"):
            _last_non_downsample_funnel(stats)

    def test_single_non_downsample_entry(self):
        """Should work with a single non-downsample entry."""
        stats = {
            "filters": [
                {"name": "raw", "type": "raw", "funnel": 42},
            ]
        }
        result = _last_non_downsample_funnel(stats)
        assert result == 42


# ──────────────────────── _count_bases_in_interval_list ──────────────────────


class TestCountBasesInIntervalList:
    """Tests for _count_bases_in_interval_list."""

    def test_basic_counting(self, tmp_path):
        """Should count bases correctly from a simple interval list."""
        interval_file = tmp_path / "test.interval_list"
        # Write a simple interval_list: 1-based closed coordinates
        # Each interval contributes end - start + 1 bases
        content = (
            "@SQ\tSN:chr1\tLN:1000\n"
            "chr1\t1\t100\t+\tinterval1\n"  # 100 bases
            "chr1\t200\t300\t+\tinterval2\n"  # 101 bases
        )
        interval_file.write_text(content)
        result = _count_bases_in_interval_list(str(interval_file))
        assert result == 100 + 101  # 201 total

    def test_skips_header_lines(self, tmp_path):
        """Should skip @-prefixed and #-prefixed header lines."""
        interval_file = tmp_path / "test.interval_list"
        content = "@HD\tVN:1.6\n@SQ\tSN:chr1\tLN:1000\n# comment line\nchr1\t10\t20\t+\tinterval1\n"  # 11 bases
        interval_file.write_text(content)
        result = _count_bases_in_interval_list(str(interval_file))
        assert result == 11

    def test_skips_empty_lines(self, tmp_path):
        """Should skip empty lines."""
        interval_file = tmp_path / "test.interval_list"
        content = "chr1\t1\t50\t+\tinterval1\n\nchr1\t100\t150\t+\tinterval2\n"
        interval_file.write_text(content)
        result = _count_bases_in_interval_list(str(interval_file))
        assert result == 50 + 51  # 101 total

    def test_file_not_found_raises(self):
        """Should raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            _count_bases_in_interval_list("/nonexistent/path/file.interval_list")

    def test_gzipped_file(self, tmp_path):
        """Should handle gzipped interval list files."""
        interval_file = tmp_path / "test.interval_list.gz"
        content = "chr1\t1\t100\t+\tinterval1\n"
        with gzip.open(interval_file, "wt", encoding="utf-8") as fh:
            fh.write(content)
        result = _count_bases_in_interval_list(str(interval_file))
        assert result == 100

    def test_fewer_than_three_fields_skipped(self, tmp_path):
        """Lines with fewer than 3 fields should be skipped."""
        interval_file = tmp_path / "test.interval_list"
        content = (
            "chr1\t1\n"  # only 2 fields, should be skipped
            "chr1\t10\t20\t+\tinterval1\n"  # 11 bases
        )
        interval_file.write_text(content)
        result = _count_bases_in_interval_list(str(interval_file))
        assert result == 11


# ──────────────────────── _configure_xgb_device ──────────────────────


class TestConfigureXgbDevice:
    """Tests for _configure_xgb_device."""

    def test_cpu_config(self):
        """Should set device=cpu and n_jobs=-1."""
        params = {}
        _configure_xgb_device(params, use_gpu=False)
        assert params["device"] == "cpu"
        assert params["n_jobs"] == -1
        assert "nthread" not in params

    def test_cpu_preserves_n_jobs(self):
        """Should not override existing n_jobs when using CPU."""
        params = {"n_jobs": 4}
        _configure_xgb_device(params, use_gpu=False)
        assert params["device"] == "cpu"
        assert params["n_jobs"] == 4

    def test_gpu_config(self):
        """Should set device=cuda and sampling_method for GPU."""
        params = {"nthread": 8, "n_jobs": 4}
        _configure_xgb_device(params, use_gpu=True)
        assert params["device"] == "cuda"
        assert params["sampling_method"] == "gradient_based"
        assert "nthread" not in params
        assert "n_jobs" not in params

    def test_cpu_removes_nthread(self):
        """Should remove nthread when using CPU."""
        params = {"nthread": 8}
        _configure_xgb_device(params, use_gpu=False)
        assert "nthread" not in params


# ──────────────────────── _parse_user_metadata ──────────────────────


class TestParseUserMetadata:
    """Tests for _parse_user_metadata."""

    def test_none_returns_empty(self):
        assert _parse_user_metadata(None) == {}

    def test_empty_list_returns_empty(self):
        assert _parse_user_metadata([]) == {}

    def test_single_token(self):
        result = _parse_user_metadata(["key=value"])
        assert result == {"key": "value"}

    def test_multiple_tokens(self):
        result = _parse_user_metadata(["adapter_version=v2", "docker_image=img:latest"])
        assert result == {"adapter_version": "v2", "docker_image": "img:latest"}

    def test_invalid_token_no_equals_raises(self):
        """Tokens without '=' should raise ValueError."""
        with pytest.raises(ValueError, match="must contain exactly one '='"):
            _parse_user_metadata(["no_equals_sign"])

    def test_too_many_equals_raises(self):
        """Tokens with more than one '=' should raise ValueError."""
        with pytest.raises(ValueError, match="must contain exactly one '='"):
            _parse_user_metadata(["a=b=c"])


# ──────────────────────── partition_into_folds ──────────────────────


class TestPartitionIntoFolds:
    """Tests for partition_into_folds."""

    def test_basic_partitioning(self):
        """Should partition chromosomes into balanced folds."""
        sizes = pd.Series(
            {"chr1": 100, "chr2": 80, "chr3": 60, "chr4": 40, "chr5": 20},
        )
        result = partition_into_folds(sizes, k_folds=2, n_chroms_leave_out=1)
        # chr5 (smallest) should be excluded
        assert "chr5" not in result
        # All remaining chroms should have fold assignments
        assert set(result.keys()) == {"chr1", "chr2", "chr3", "chr4"}
        assert all(v in (0, 1) for v in result.values())

    def test_single_fold(self):
        """With k_folds=1, all should go to fold 0."""
        sizes = pd.Series({"chr1": 100, "chr2": 50, "chr3": 30})
        result = partition_into_folds(sizes, k_folds=1, n_chroms_leave_out=1)
        # Smallest (chr3) excluded
        assert "chr3" not in result
        assert all(v == 0 for v in result.values())

    def test_no_chroms_leave_out(self):
        """With n_chroms_leave_out=0, all chromosomes should be assigned."""
        sizes = pd.Series({"chr1": 100, "chr2": 50})
        result = partition_into_folds(sizes, k_folds=2, n_chroms_leave_out=0)
        assert set(result.keys()) == {"chr1", "chr2"}

    def test_balanced_result(self):
        """Greedy algorithm should produce roughly balanced folds."""
        sizes = pd.Series({f"chr{i}": (22 - i) * 10 for i in range(1, 11)})
        result = partition_into_folds(sizes, k_folds=3, n_chroms_leave_out=1)
        # Count total size per fold
        fold_sizes = [0, 0, 0]
        for chrom, fold in result.items():
            fold_sizes[fold] += sizes[chrom]
        # Max fold should not be more than 2x min fold
        assert max(fold_sizes) / max(min(fold_sizes), 1) < 2.0

    def test_invalid_algorithm_raises(self):
        """Should raise ValueError for unsupported algorithm."""
        sizes = pd.Series({"chr1": 100})
        with pytest.raises(ValueError, match="Only greedy algorithm"):
            partition_into_folds(sizes, k_folds=2, alg="random")


# ──────────────────────── _probability_recalibration ──────────────────────


class TestProbabilityRecalibration:
    """Tests for _probability_recalibration (identity mapping)."""

    def test_returns_copy_of_input(self):
        """Should return an identical copy of the probabilities."""
        prob_orig = np.array([0.1, 0.5, 0.9])
        y_all = np.array([0, 1, 1])
        result = _probability_recalibration(prob_orig, y_all)
        np.testing.assert_array_equal(result, prob_orig)

    def test_returns_new_array(self):
        """Should return a new array, not the same object."""
        prob_orig = np.array([0.1, 0.5, 0.9])
        y_all = np.array([0, 1, 1])
        result = _probability_recalibration(prob_orig, y_all)
        assert result is not prob_orig

    def test_modifications_dont_affect_original(self):
        """Modifying result should not affect original."""
        prob_orig = np.array([0.1, 0.5, 0.9])
        y_all = np.array([0, 1, 1])
        result = _probability_recalibration(prob_orig, y_all)
        result[0] = 999.0
        assert prob_orig[0] == 0.1


# ══════════════════════════════════════════════════════════════════════════════
# SRSNVTrainer class method tests (mocked)
# ══════════════════════════════════════════════════════════════════════════════


def _make_synthetic_pos_df(n_rows=100):
    """Build a synthetic positive DataFrame matching expected columns."""
    rng = np.random.default_rng(42)
    return pl.DataFrame(
        {
            CHROM: ["chr1"] * n_rows,
            POS: list(range(1, n_rows + 1)),
            REF: pl.Series(["A"] * n_rows).cast(pl.Categorical),
            X_ALT: pl.Series(rng.choice(["C", "G", "T"], n_rows).tolist()).cast(pl.Categorical),
            X_HMER_REF: rng.integers(1, 10, n_rows).tolist(),
            X_HMER_ALT: rng.integers(1, 10, n_rows).tolist(),
            FeatureMapFields.EDIST.value: rng.integers(0, 5, n_rows).tolist(),
            FeatureMapFields.HAMDIST.value: rng.integers(0, 5, n_rows).tolist(),
            "RN": [f"read_{i}" for i in range(n_rows)],
            MQUAL: rng.uniform(0, 60, n_rows).tolist(),
        }
    )


def _make_synthetic_neg_df(n_rows=200):
    """Build a synthetic negative DataFrame matching expected columns."""
    rng = np.random.default_rng(99)
    return pl.DataFrame(
        {
            CHROM: ["chr1"] * n_rows,
            POS: list(range(1, n_rows + 1)),
            REF: pl.Series(rng.choice(["A", "C", "G", "T"], n_rows).tolist()).cast(pl.Categorical),
            X_ALT: pl.Series(rng.choice(["A", "C", "G", "T"], n_rows).tolist()).cast(pl.Categorical),
            X_HMER_REF: rng.integers(1, 10, n_rows).tolist(),
            X_HMER_ALT: rng.integers(1, 10, n_rows).tolist(),
            FeatureMapFields.EDIST.value: rng.integers(0, 5, n_rows).tolist(),
            FeatureMapFields.HAMDIST.value: rng.integers(0, 5, n_rows).tolist(),
            "RN": [f"read_{i}" for i in range(n_rows)],
            MQUAL: rng.uniform(0, 60, n_rows).tolist(),
        }
    )


def _make_unified_stats():
    """Build a minimal unified stats dict for testing."""
    return {
        FILTERS_RANDOM_SAMPLE: {
            "filters": {
                "raw": {"funnel": 5000, "pass": 5000},
                "quality_filter": {"type": "quality", "threshold": 30, "funnel": 3000, "pass": 3000},
                "label_filter": {"type": "label", "funnel": 1000, "pass": 1000},
            },
        },
        FILTERS_FULL_OUTPUT: {
            "filters": {
                "raw": {"funnel": 100000, "pass": 100000},
                "quality_filter": {"type": "quality", "threshold": 30, "funnel": 80000, "pass": 80000},
                "region_filter": {"type": "region", "bed": "test.bed", "funnel": 50000, "pass": 50000},
            },
        },
    }


def _make_mock_args(tmp_path):
    """Create a mock argparse.Namespace with all required fields."""
    return argparse.Namespace(
        positive="/fake/positive.parquet",
        negative="/fake/negative.parquet",
        stats_file="/fake/stats.json",
        mean_coverage=30.0,
        training_regions="/fake/regions.interval_list",
        k_folds=1,
        model_params=None,
        features=None,
        output=str(tmp_path),
        basename="test.",
        random_seed=42,
        verbose=False,
        max_qual=100.0,
        quality_lut_size=100,
        metadata=None,
        use_gpu=False,
        use_float32=False,
        use_kde_smoothing=False,
        single_model_split=False,
        holdout_chromosomes=None,
        val_chromosomes=None,
        val_fraction=0.1,
        split_hash_key="RN",
        split_manifest_in=None,
        split_manifest_out=None,
    )


# ──────────────────────── _read_positive_df ──────────────────────


class TestReadPositiveDf:
    """Tests for SRSNVTrainer._read_positive_df with mocked parquet reading."""

    def _make_trainer_stub(self):
        """Create a minimal trainer object for testing _read_positive_df."""
        trainer = object.__new__(SRSNVTrainer)
        return trainer

    def test_ref_replaced_by_x_alt(self):
        """REF column should be replaced by X_ALT values."""
        trainer = self._make_trainer_stub()
        # Use a df without EDIST so no rows get filtered
        input_df = pl.DataFrame(
            {
                CHROM: ["chr1"] * 5,
                POS: [1, 2, 3, 4, 5],
                REF: pl.Series(["A"] * 5).cast(pl.Categorical),
                X_ALT: pl.Series(["C", "G", "T", "C", "G"]).cast(pl.Categorical),
                X_HMER_REF: [1, 2, 3, 4, 5],
                X_HMER_ALT: [5, 4, 3, 2, 1],
                MQUAL: [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        original_x_alt = input_df[X_ALT].to_list()

        with patch("ugbio_srsnv.srsnv_training.pl.read_parquet", return_value=input_df):
            result = trainer._read_positive_df("/fake/pos.parquet")

        # X_ALT should be dropped
        assert X_ALT not in result.columns
        # REF should contain what was in X_ALT
        assert result[REF].to_list() == original_x_alt

    def test_hmer_columns_swapped(self):
        """X_HMER_REF and X_HMER_ALT should be swapped in positive set."""
        trainer = self._make_trainer_stub()
        # Use a df without EDIST so no rows get filtered
        input_df = pl.DataFrame(
            {
                CHROM: ["chr1"] * 4,
                POS: [1, 2, 3, 4],
                REF: pl.Series(["A"] * 4).cast(pl.Categorical),
                X_ALT: pl.Series(["C"] * 4).cast(pl.Categorical),
                X_HMER_REF: [1, 2, 3, 4],
                X_HMER_ALT: [10, 20, 30, 40],
                MQUAL: [10.0, 20.0, 30.0, 40.0],
            }
        )

        with patch("ugbio_srsnv.srsnv_training.pl.read_parquet", return_value=input_df):
            result = trainer._read_positive_df("/fake/pos.parquet")

        # After swap: X_HMER_REF should have old X_HMER_ALT values
        assert result[X_HMER_REF].to_list() == [10, 20, 30, 40]
        assert result[X_HMER_ALT].to_list() == [1, 2, 3, 4]

    def test_edist_max_rows_removed(self):
        """Rows where EDIST == max(EDIST) should be removed."""
        trainer = self._make_trainer_stub()
        # Create a small df with known EDIST values
        edist_col = FeatureMapFields.EDIST.value
        input_df = pl.DataFrame(
            {
                CHROM: ["chr1"] * 5,
                POS: [1, 2, 3, 4, 5],
                REF: pl.Series(["A"] * 5).cast(pl.Categorical),
                X_ALT: pl.Series(["C"] * 5).cast(pl.Categorical),
                X_HMER_REF: [1, 2, 3, 4, 5],
                X_HMER_ALT: [5, 4, 3, 2, 1],
                edist_col: [0, 1, 2, 3, 4],  # max is 4
                MQUAL: [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )

        with patch("ugbio_srsnv.srsnv_training.pl.read_parquet", return_value=input_df):
            result = trainer._read_positive_df("/fake/pos.parquet")

        # Row with EDIST=4 (max) should be removed, leaving 4 rows
        assert result.height == 4
        # After removal, remaining EDIST values [0,1,2,3] get incremented to [1,2,3,4]
        # The key point is 1 row was removed (the one with max EDIST=4)
        # The original values [0,1,2,3] become [1,2,3,4] after +1 increment
        assert result[edist_col].to_list() == [1, 2, 3, 4]

    def test_edist_features_incremented(self):
        """EDIST features should be incremented by 1."""
        trainer = self._make_trainer_stub()
        edist_col = FeatureMapFields.EDIST.value
        hamdist_col = FeatureMapFields.HAMDIST.value
        input_df = pl.DataFrame(
            {
                CHROM: ["chr1"] * 4,
                POS: [1, 2, 3, 4],
                REF: pl.Series(["A"] * 4).cast(pl.Categorical),
                X_ALT: pl.Series(["C"] * 4).cast(pl.Categorical),
                X_HMER_REF: [1, 2, 3, 4],
                X_HMER_ALT: [4, 3, 2, 1],
                edist_col: [0, 1, 2, 3],  # max=3, row with 3 removed
                hamdist_col: [0, 1, 2, 3],
                MQUAL: [10.0, 20.0, 30.0, 40.0],
            }
        )

        with patch("ugbio_srsnv.srsnv_training.pl.read_parquet", return_value=input_df):
            result = trainer._read_positive_df("/fake/pos.parquet")

        # Rows with edist=3 (max) removed, remaining are [0,1,2] which become [1,2,3]
        assert result[edist_col].to_list() == [1, 2, 3]
        assert result[hamdist_col].to_list() == [1, 2, 3]

    def test_label_column_added_true(self):
        """Label column should be True for all positive rows."""
        trainer = self._make_trainer_stub()
        pos_df = _make_synthetic_pos_df(20)

        with patch("ugbio_srsnv.srsnv_training.pl.read_parquet", return_value=pos_df):
            result = trainer._read_positive_df("/fake/pos.parquet")

        assert LABEL_COL in result.columns
        assert result[LABEL_COL].all()

    def test_missing_x_alt_raises(self):
        """Should raise ValueError if X_ALT column is missing."""
        trainer = self._make_trainer_stub()
        input_df = pl.DataFrame(
            {
                CHROM: ["chr1"],
                POS: [1],
                REF: pl.Series(["A"]).cast(pl.Categorical),
                X_HMER_REF: [1],
                X_HMER_ALT: [2],
                MQUAL: [30.0],
            }
        )

        with patch("ugbio_srsnv.srsnv_training.pl.read_parquet", return_value=input_df):
            with pytest.raises(ValueError, match="missing required column 'X_ALT'"):
                trainer._read_positive_df("/fake/pos.parquet")


# ──────────────────────── _read_negative_df ──────────────────────


class TestReadNegativeDf:
    """Tests for SRSNVTrainer._read_negative_df with mocked parquet reading."""

    def _make_trainer_stub(self):
        """Create a minimal trainer object for testing _read_negative_df."""
        return object.__new__(SRSNVTrainer)

    def test_label_column_added_false(self):
        """Label column should be False for all negative rows."""
        trainer = self._make_trainer_stub()
        neg_df = _make_synthetic_neg_df(30)

        with patch("ugbio_srsnv.srsnv_training.pl.read_parquet", return_value=neg_df):
            result = trainer._read_negative_df("/fake/neg.parquet")

        assert LABEL_COL in result.columns
        assert not result[LABEL_COL].any()

    def test_x_alt_dropped(self):
        """X_ALT column should be dropped from negative dataframe."""
        trainer = self._make_trainer_stub()
        neg_df = _make_synthetic_neg_df(30)
        assert X_ALT in neg_df.columns  # precondition

        with patch("ugbio_srsnv.srsnv_training.pl.read_parquet", return_value=neg_df):
            result = trainer._read_negative_df("/fake/neg.parquet")

        assert X_ALT not in result.columns

    def test_no_hmer_swap(self):
        """Negative set should NOT swap X_HMER_REF and X_HMER_ALT."""
        trainer = self._make_trainer_stub()
        neg_df = _make_synthetic_neg_df(30)
        original_hmer_ref = neg_df[X_HMER_REF].to_list()
        original_hmer_alt = neg_df[X_HMER_ALT].to_list()

        with patch("ugbio_srsnv.srsnv_training.pl.read_parquet", return_value=neg_df):
            result = trainer._read_negative_df("/fake/neg.parquet")

        assert result[X_HMER_REF].to_list() == original_hmer_ref
        assert result[X_HMER_ALT].to_list() == original_hmer_alt


# ──────────────────────── _load_data ──────────────────────


class TestLoadData:
    """Tests for SRSNVTrainer._load_data."""

    def _make_trainer_stub(self):
        return object.__new__(SRSNVTrainer)

    def test_concatenates_pos_and_neg(self):
        """Should concatenate positive and negative dataframes."""
        trainer = self._make_trainer_stub()
        # Make compatible pos and neg (same columns after processing)
        edist_col = FeatureMapFields.EDIST.value
        hamdist_col = FeatureMapFields.HAMDIST.value
        cols = {
            CHROM: ["chr1"] * 3,
            POS: [1, 2, 3],
            REF: pl.Series(["A"] * 3).cast(pl.Categorical),
            X_HMER_REF: [1, 2, 3],
            X_HMER_ALT: [3, 2, 1],
            edist_col: [0, 1, 2],
            hamdist_col: [0, 1, 2],
            MQUAL: [10.0, 20.0, 30.0],
            "RN": ["r1", "r2", "r3"],
        }
        pos_result = pl.DataFrame(cols).with_columns(pl.lit(value=True).alias(LABEL_COL))
        neg_result = pl.DataFrame(cols).with_columns(pl.lit(value=False).alias(LABEL_COL))

        with (
            patch.object(trainer, "_read_positive_df", return_value=pos_result),
            patch.object(trainer, "_read_negative_df", return_value=neg_result),
        ):
            result = trainer._load_data("/fake/pos.parquet", "/fake/neg.parquet")

        assert result.height == 6  # 3 + 3

    def test_column_mismatch_raises(self):
        """Should raise ValueError if columns don't match."""
        trainer = self._make_trainer_stub()
        pos_result = pl.DataFrame(
            {
                CHROM: ["chr1"],
                POS: [1],
                REF: pl.Series(["A"]).cast(pl.Categorical),
                LABEL_COL: [True],
                "EXTRA_COL": [1],
            }
        )
        neg_result = pl.DataFrame(
            {
                CHROM: ["chr1"],
                POS: [1],
                REF: pl.Series(["A"]).cast(pl.Categorical),
                LABEL_COL: [False],
            }
        )

        with (
            patch.object(trainer, "_read_positive_df", return_value=pos_result),
            patch.object(trainer, "_read_negative_df", return_value=neg_result),
        ):
            with pytest.raises(ValueError, match="different columns"):
                trainer._load_data("/fake/pos.parquet", "/fake/neg.parquet")


# ──────────────────────── _feature_columns ──────────────────────


class TestFeatureColumns:
    """Tests for SRSNVTrainer._feature_columns."""

    def _make_trainer_with_df(self, columns, feature_list=None):
        """Create a trainer with a mocked data_frame."""
        trainer = object.__new__(SRSNVTrainer)
        # Build a simple DataFrame with the given columns
        data = {col: [1] for col in columns}
        trainer.data_frame = pl.DataFrame(data)
        trainer.feature_list = feature_list
        return trainer

    def test_excludes_label_fold_chrom_pos(self):
        """Should exclude LABEL, FOLD, CHROM, POS from features."""
        columns = [LABEL_COL, FOLD_COL, CHROM, POS, "feat_A", "feat_B", MQUAL]
        trainer = self._make_trainer_with_df(columns)
        result = trainer._feature_columns()
        assert LABEL_COL not in result
        assert FOLD_COL not in result
        assert CHROM not in result
        assert POS not in result
        assert set(result) == {"feat_A", "feat_B", MQUAL}

    def test_user_feature_list_intersection(self):
        """With a user feature list, should return intersection only."""
        columns = [LABEL_COL, FOLD_COL, CHROM, POS, "feat_A", "feat_B", "feat_C"]
        trainer = self._make_trainer_with_df(columns, feature_list=["feat_A", "feat_C"])
        result = trainer._feature_columns()
        assert result == ["feat_A", "feat_C"]

    def test_user_feature_missing_raises(self):
        """Should raise ValueError if user-specified feature is missing."""
        columns = [LABEL_COL, FOLD_COL, CHROM, POS, "feat_A"]
        trainer = self._make_trainer_with_df(columns, feature_list=["feat_A", "feat_MISSING"])
        with pytest.raises(ValueError, match="Requested feature.*absent from data"):
            trainer._feature_columns()

    def test_all_features_returned_when_no_filter(self):
        """Without feature_list, all non-excluded columns should be returned."""
        columns = [LABEL_COL, FOLD_COL, CHROM, POS, "X_HMER_REF", "X_HMER_ALT", "MQUAL", "RN"]
        trainer = self._make_trainer_with_df(columns)
        result = trainer._feature_columns()
        assert set(result) == {"X_HMER_REF", "X_HMER_ALT", "MQUAL", "RN"}


# ──────────────────────── _extract_categorical_encodings / _extract_feature_dtypes ──────


class TestExtractEncodingsAndDtypes:
    """Tests for _extract_categorical_encodings and _extract_feature_dtypes."""

    def _make_trainer_stub(self):
        trainer = object.__new__(SRSNVTrainer)
        trainer.categorical_encodings = {}
        trainer.feature_dtypes = {}
        return trainer

    def test_extract_categorical_encodings(self):
        """Should detect categorical columns and build encoding dict."""
        trainer = self._make_trainer_stub()
        pd_df = pd.DataFrame(
            {
                "cat_col": pd.Categorical(["A", "B", "C", "A", "B"]),
                "num_col": [1.0, 2.0, 3.0, 4.0, 5.0],
                "int_col": [10, 20, 30, 40, 50],
            }
        )
        feat_cols = ["cat_col", "num_col", "int_col"]
        trainer._extract_categorical_encodings(pd_df, feat_cols)

        assert "cat_col" in trainer.categorical_encodings
        assert trainer.categorical_encodings["cat_col"] == {"A": 0, "B": 1, "C": 2}
        assert "num_col" not in trainer.categorical_encodings
        assert "int_col" not in trainer.categorical_encodings

    def test_extract_feature_dtypes(self):
        """Should record dtype strings for all feature columns."""
        trainer = self._make_trainer_stub()
        pd_df = pd.DataFrame(
            {
                "cat_col": pd.Categorical(["A", "B", "C"]),
                "float_col": [1.0, 2.0, 3.0],
                "int_col": pd.array([1, 2, 3], dtype="int64"),
            }
        )
        feat_cols = ["cat_col", "float_col", "int_col"]
        trainer._extract_feature_dtypes(pd_df, feat_cols)

        assert "cat_col" in trainer.feature_dtypes
        assert trainer.feature_dtypes["cat_col"] == "category"
        assert trainer.feature_dtypes["float_col"] == "float64"
        assert trainer.feature_dtypes["int_col"] == "int64"


# ──────────────────────── _validate_kde_args ──────────────────────


class TestValidateKdeArgs:
    """Tests for SRSNVTrainer._validate_kde_args."""

    def _make_trainer_stub(self):
        return object.__new__(SRSNVTrainer)

    def test_valid_args(self):
        """Should not raise for valid arguments."""
        trainer = self._make_trainer_stub()
        trainer._validate_kde_args("mqual", "fp")
        trainer._validate_kde_args("logit", "tp")
        trainer._validate_kde_args("logit", "mp")

    def test_invalid_transform_mode(self):
        """Should raise for invalid transform_mode."""
        trainer = self._make_trainer_stub()
        with pytest.raises(ValueError, match="transform_mode must be one of"):
            trainer._validate_kde_args("invalid", "fp")

    def test_invalid_mqual_cutoff_type(self):
        """Should raise for invalid mqual_cutoff_type."""
        trainer = self._make_trainer_stub()
        with pytest.raises(ValueError, match="mqual_cutoff_type must be one of"):
            trainer._validate_kde_args("mqual", "invalid")


# ──────────────────────── _calculate_snvq_prefactor ──────────────────────


class TestCalculateSnvqPrefactor:
    """Tests for SRSNVTrainer._calculate_snvq_prefactor."""

    def _make_trainer_stub(self):
        trainer = object.__new__(SRSNVTrainer)
        # Set up pos_stats with proper filter format for get_filter_ratio
        trainer.pos_stats = {
            "filters": [
                {"name": "raw", "type": "raw", "funnel": 10000},
                {"name": "quality", "type": "quality", "funnel": 8000},
                {"name": "label_filter", "type": "label", "funnel": 5000},
            ]
        }
        trainer.mean_coverage = 30.0
        trainer.n_bases_in_region = 1000000
        trainer.raw_featuremap_size_filtered = 50000
        return trainer

    def test_calculates_prefactor(self):
        """Should calculate snvq_prefactor = raw_featuremap_size_filtered / effective_bases_covered."""
        trainer = self._make_trainer_stub()
        result = trainer._calculate_snvq_prefactor()

        # get_filter_ratio with numerator_type="label", denominator_type="raw" should return
        # rows before the "label" filter / rows at "raw" = 8000 / 10000 = 0.8
        filtering_ratio = 8000 / 10000
        expected_effective_bases = 30.0 * 1000000 * filtering_ratio
        expected_prefactor = 50000 / expected_effective_bases

        assert abs(result - expected_prefactor) < 1e-10
        assert abs(trainer.snvq_prefactor - expected_prefactor) < 1e-10
        assert abs(trainer.effective_bases_covered - expected_effective_bases) < 1e-10


# ──────────────────────── _init_models ──────────────────────


class TestInitModels:
    """Tests for SRSNVTrainer._init_models."""

    def _make_trainer_stub(self, k_folds=2):
        trainer = object.__new__(SRSNVTrainer)
        trainer.k_folds = k_folds
        trainer.use_gpu = False
        return trainer

    def test_creates_correct_number_of_models(self):
        """Should create k_folds models."""
        trainer = self._make_trainer_stub(k_folds=3)
        args = argparse.Namespace(model_params=None)
        trainer._init_models(args)
        assert len(trainer.models) == 3

    def test_model_params_applied(self):
        """Should parse and apply model parameters."""
        trainer = self._make_trainer_stub(k_folds=1)
        args = argparse.Namespace(model_params="eta=0.1:max_depth=6")
        trainer._init_models(args)
        assert len(trainer.models) == 1
        # Check that params were stored
        assert trainer.model_params["eta"] == 0.1
        assert trainer.model_params["max_depth"] == 6

    def test_default_early_stopping_and_n_estimators(self):
        """Should set default early_stopping_rounds and n_estimators if not specified."""
        trainer = self._make_trainer_stub(k_folds=1)
        args = argparse.Namespace(model_params=None)
        trainer._init_models(args)
        assert trainer.model_params["early_stopping_rounds"] == 10
        assert trainer.model_params["n_estimators"] == 2000

    def test_cpu_device_configured(self):
        """Should configure device=cpu when not using GPU."""
        trainer = self._make_trainer_stub(k_folds=1)
        args = argparse.Namespace(model_params=None)
        trainer._init_models(args)
        assert trainer.model_params["device"] == "cpu"
        assert trainer.model_params["n_jobs"] == -1


# ──────────────────────── _build_new_split_manifest ──────────────────────


class TestBuildNewSplitManifest:
    """Tests for _build_new_split_manifest standalone function."""

    def test_calls_build_split_manifest_for_kfold(self):
        """Should call build_split_manifest when not single_model_split."""
        args = argparse.Namespace(training_regions="/fake/regions.interval_list")
        expected = {"split_mode": "chromosome_kfold", "chrom_to_fold": {"chr1": 0}}

        with patch("ugbio_srsnv.srsnv_training.build_split_manifest", return_value=expected) as mock_build:
            result = _build_new_split_manifest(
                args=args,
                single_model_split=False,
                val_chromosomes=[],
                holdout_chromosomes=None,
                seed=42,
                k_folds=3,
                val_fraction=0.1,
                split_hash_key="RN",
            )

        mock_build.assert_called_once()
        assert result == expected

    def test_calls_single_model_chrom_val_manifest(self):
        """Should call build_single_model_chrom_val_manifest when val_chromosomes provided."""
        args = argparse.Namespace(training_regions="/fake/regions.interval_list")
        expected = {"split_mode": "single_model_chrom_val"}

        with patch(
            "ugbio_srsnv.srsnv_training.build_single_model_chrom_val_manifest", return_value=expected
        ) as mock_build:
            result = _build_new_split_manifest(
                args=args,
                single_model_split=True,
                val_chromosomes=["chr20"],
                holdout_chromosomes=["chr21", "chr22"],
                seed=42,
                k_folds=1,
                val_fraction=0.1,
                split_hash_key="RN",
            )

        mock_build.assert_called_once_with(
            training_regions="/fake/regions.interval_list",
            holdout_chromosomes=["chr21", "chr22"],
            val_chromosomes=["chr20"],
        )
        assert result == expected

    def test_calls_single_model_read_hash_manifest(self):
        """Should call build_single_model_read_hash_manifest when no val_chromosomes."""
        args = argparse.Namespace(training_regions="/fake/regions.interval_list")
        expected = {"split_mode": "single_model_read_hash"}

        with patch(
            "ugbio_srsnv.srsnv_training.build_single_model_read_hash_manifest", return_value=expected
        ) as mock_build:
            result = _build_new_split_manifest(
                args=args,
                single_model_split=True,
                val_chromosomes=[],
                holdout_chromosomes=["chr21", "chr22"],
                seed=42,
                k_folds=1,
                val_fraction=0.15,
                split_hash_key="RN",
            )

        mock_build.assert_called_once_with(
            training_regions="/fake/regions.interval_list",
            random_seed=42,
            holdout_chromosomes=["chr21", "chr22"],
            val_fraction=0.15,
            hash_key="RN",
        )
        assert result == expected


# ──────────────────────── save ──────────────────────


class TestSave:
    """Tests for SRSNVTrainer.save."""

    def _make_trainer_for_save(self, tmp_path):
        """Create a trainer with enough state to call save()."""
        trainer = object.__new__(SRSNVTrainer)
        trainer.args = argparse.Namespace(basename="test.")
        trainer.out_dir = tmp_path
        trainer.seed = 42
        trainer.max_qual = 100.0
        trainer.k_folds = 1
        trainer.chrom_to_fold = {"chr1": 0, "chr2": 0}
        trainer.user_metadata = {"version": "1.0"}
        trainer.model_params = {"eta": 0.1, "device": "cpu"}

        # Create a small data_frame
        trainer.data_frame = pl.DataFrame(
            {
                CHROM: ["chr1", "chr2", "chr1"],
                POS: [1, 2, 3],
                LABEL_COL: [True, True, False],
                FOLD_COL: [0, 0, 0],
                MQUAL: [30.0, 40.0, 20.0],
            }
        )
        trainer.n_neg = 1

        # Stats
        trainer.pos_stats = {
            "filters": [
                {"name": "raw", "type": "raw", "funnel": 5000},
                {"name": "quality", "type": "quality", "funnel": 3000},
            ]
        }
        trainer.neg_stats = {
            "filters": [
                {"name": "raw", "type": "raw", "funnel": 100000},
                {"name": "region", "type": "region", "funnel": 50000},
            ]
        }
        trainer.prior_train_error = 1 / 3

        # Quality lookup table
        trainer.x_lut = np.linspace(0, 60, 100)
        trainer.y_lut = np.linspace(0, 100, 100)

        # Feature metadata
        trainer.feature_dtypes = {"MQUAL": "float64"}
        trainer.categorical_encodings = {}
        trainer.effective_bases_covered = 1e8
        trainer.snvq_prefactor = 0.5

        # Training results
        trainer.training_results = [{"validation_0": {"auc": [0.9]}}]

        # Mock model
        mock_model = MagicMock()
        mock_model.save_model = MagicMock()
        trainer.models = [mock_model]

        return trainer

    def test_saves_metadata_json(self, tmp_path):
        """Should save metadata JSON with expected structure."""
        trainer = self._make_trainer_for_save(tmp_path)
        trainer.save()

        metadata_path = tmp_path / "test.srsnv_metadata.json"
        assert metadata_path.is_file()

        with metadata_path.open() as fh:
            metadata = json.load(fh)

        assert "model_paths" in metadata
        assert "features" in metadata
        assert "quality_recalibration_table" in metadata
        assert "filtering_stats" in metadata
        assert "model_params" in metadata
        assert "training_parameters" in metadata
        assert "metadata" in metadata
        assert metadata["metadata"] == {"version": "1.0"}

    def test_saves_parquet(self, tmp_path):
        """Should save dataframe as parquet."""
        trainer = self._make_trainer_for_save(tmp_path)
        trainer.save()

        df_path = tmp_path / "test.featuremap_df.parquet"
        assert df_path.is_file()

    def test_downsample_segments_in_stats(self, tmp_path):
        """Should add downsample segments to pos and neg stats."""
        trainer = self._make_trainer_for_save(tmp_path)
        trainer.save()

        metadata_path = tmp_path / "test.srsnv_metadata.json"
        with metadata_path.open() as fh:
            metadata = json.load(fh)

        pos_filters = metadata["filtering_stats"]["positive"]["filters"]
        neg_filters = metadata["filtering_stats"]["negative"]["filters"]

        # Last filter should be downsample type
        assert pos_filters[-1]["type"] == "downsample"
        assert neg_filters[-1]["type"] == "downsample"
        assert pos_filters[-1]["funnel"] == 2  # n_pos = 3 - 1 = 2
        assert neg_filters[-1]["funnel"] == 1  # n_neg = 1

    def test_quality_recalibration_table_format(self, tmp_path):
        """Quality recalibration table should be [x_lut_list, y_lut_list]."""
        trainer = self._make_trainer_for_save(tmp_path)
        trainer.save()

        metadata_path = tmp_path / "test.srsnv_metadata.json"
        with metadata_path.open() as fh:
            metadata = json.load(fh)

        qrt = metadata["quality_recalibration_table"]
        assert len(qrt) == 2
        assert len(qrt[0]) == 100  # x_lut
        assert len(qrt[1]) == 100  # y_lut


# ──────────────────────── Full SRSNVTrainer.__init__ (mocked) ──────────────────────


class TestSRSNVTrainerInit:
    """Tests for SRSNVTrainer.__init__ with all dependencies mocked."""

    def _make_pos_neg_dfs(self):
        """Create compatible pos and neg DataFrames for full init test."""
        edist_col = FeatureMapFields.EDIST.value
        hamdist_col = FeatureMapFields.HAMDIST.value
        rng = np.random.default_rng(42)
        n_pos, n_neg = 50, 100

        pos_df = pl.DataFrame(
            {
                CHROM: ["chr1"] * 30 + ["chr2"] * 20,
                POS: list(range(1, n_pos + 1)),
                REF: pl.Series(["A"] * n_pos).cast(pl.Categorical),
                X_ALT: pl.Series(rng.choice(["C", "G", "T"], n_pos).tolist()).cast(pl.Categorical),
                X_HMER_REF: rng.integers(1, 10, n_pos).tolist(),
                X_HMER_ALT: rng.integers(1, 10, n_pos).tolist(),
                edist_col: rng.integers(0, 4, n_pos).tolist(),
                hamdist_col: rng.integers(0, 4, n_pos).tolist(),
                MQUAL: rng.uniform(0, 60, n_pos).tolist(),
                "RN": [f"read_p{i}" for i in range(n_pos)],
            }
        )

        neg_df = pl.DataFrame(
            {
                CHROM: ["chr1"] * 60 + ["chr2"] * 40,
                POS: list(range(1, n_neg + 1)),
                REF: pl.Series(rng.choice(["A", "C", "G", "T"], n_neg).tolist()).cast(pl.Categorical),
                X_ALT: pl.Series(rng.choice(["A", "C", "G", "T"], n_neg).tolist()).cast(pl.Categorical),
                X_HMER_REF: rng.integers(1, 10, n_neg).tolist(),
                X_HMER_ALT: rng.integers(1, 10, n_neg).tolist(),
                edist_col: rng.integers(0, 4, n_neg).tolist(),
                hamdist_col: rng.integers(0, 4, n_neg).tolist(),
                MQUAL: rng.uniform(0, 60, n_neg).tolist(),
                "RN": [f"read_n{i}" for i in range(n_neg)],
            }
        )
        return pos_df, neg_df

    def test_full_init_kfold(self, tmp_path):
        """Should fully initialize a trainer with k-fold split using mocks."""
        pos_df, neg_df = self._make_pos_neg_dfs()
        split_manifest = {
            "split_mode": "chromosome_kfold",
            "chrom_to_fold": {"chr1": 0, "chr2": 0},
            "k_folds": 1,
            "holdout_chromosomes": [],
        }

        args = _make_mock_args(tmp_path)

        def mock_read_parquet(path, *a, **kw):
            if "positive" in str(path) or path == args.positive:
                return pos_df
            return neg_df

        with (
            patch("ugbio_srsnv.srsnv_training.pl.read_parquet", side_effect=mock_read_parquet),
            patch("ugbio_srsnv.srsnv_training._extract_stats_from_unified") as mock_stats,
            patch("ugbio_srsnv.srsnv_training._count_bases_in_interval_list", return_value=1000000),
            patch("ugbio_srsnv.srsnv_training._validate_quality_region_filters"),
            patch("ugbio_srsnv.srsnv_training._last_non_downsample_funnel", return_value=80000),
            patch("ugbio_srsnv.srsnv_training._build_new_split_manifest", return_value=split_manifest),
        ):
            # Set up stats mock to return proper filter format
            pos_stats = {
                "filters": [
                    {"name": "raw", "type": "raw", "funnel": 10000},
                    {"name": "quality", "type": "quality", "funnel": 8000},
                    {"name": "label_filter", "type": "label", "funnel": 5000},
                ]
            }
            neg_stats = {
                "filters": [
                    {"name": "raw", "type": "raw", "funnel": 100000},
                    {"name": "region", "type": "region", "funnel": 80000},
                ]
            }
            mock_stats.return_value = (pos_stats, neg_stats)

            trainer = SRSNVTrainer(args)

        # Verify trainer state
        assert trainer.k_folds == 1
        assert trainer.mean_coverage == 30.0
        assert trainer.n_bases_in_region == 1000000
        assert trainer.data_frame is not None
        assert trainer.data_frame.height > 0
        assert LABEL_COL in trainer.data_frame.columns
        assert FOLD_COL in trainer.data_frame.columns
        assert len(trainer.models) == 1
        assert trainer.seed == 42

    def test_init_sets_output_directory(self, tmp_path):
        """Should create output directory during init."""
        pos_df, neg_df = self._make_pos_neg_dfs()
        split_manifest = {
            "split_mode": "chromosome_kfold",
            "chrom_to_fold": {"chr1": 0, "chr2": 0},
            "k_folds": 1,
            "holdout_chromosomes": [],
        }

        out_dir = tmp_path / "new_output"
        args = _make_mock_args(tmp_path)
        args.output = str(out_dir)

        def mock_read_parquet(path, *a, **kw):
            if "positive" in str(path) or path == args.positive:
                return pos_df
            return neg_df

        with (
            patch("ugbio_srsnv.srsnv_training.pl.read_parquet", side_effect=mock_read_parquet),
            patch("ugbio_srsnv.srsnv_training._extract_stats_from_unified") as mock_stats,
            patch("ugbio_srsnv.srsnv_training._count_bases_in_interval_list", return_value=1000000),
            patch("ugbio_srsnv.srsnv_training._validate_quality_region_filters"),
            patch("ugbio_srsnv.srsnv_training._last_non_downsample_funnel", return_value=80000),
            patch("ugbio_srsnv.srsnv_training._build_new_split_manifest", return_value=split_manifest),
        ):
            pos_stats = {
                "filters": [
                    {"name": "raw", "type": "raw", "funnel": 10000},
                    {"name": "quality", "type": "quality", "funnel": 8000},
                    {"name": "label_filter", "type": "label", "funnel": 5000},
                ]
            }
            neg_stats = {
                "filters": [
                    {"name": "raw", "type": "raw", "funnel": 100000},
                    {"name": "region", "type": "region", "funnel": 80000},
                ]
            }
            mock_stats.return_value = (pos_stats, neg_stats)
            SRSNVTrainer(args)

        assert out_dir.exists()

    def test_init_missing_mean_coverage_raises(self, tmp_path):
        """Should raise ValueError if mean_coverage is None."""
        args = _make_mock_args(tmp_path)
        args.mean_coverage = None

        with (
            patch("ugbio_srsnv.srsnv_training._extract_stats_from_unified") as mock_stats,
            patch("ugbio_srsnv.srsnv_training._count_bases_in_interval_list", return_value=1000000),
        ):
            pos_stats = {
                "filters": [
                    {"name": "raw", "type": "raw", "funnel": 10000},
                    {"name": "label_filter", "type": "label", "funnel": 5000},
                ]
            }
            neg_stats = {
                "filters": [
                    {"name": "raw", "type": "raw", "funnel": 100000},
                ]
            }
            mock_stats.return_value = (pos_stats, neg_stats)

            with pytest.raises(ValueError, match="--mean-coverage is required"):
                SRSNVTrainer(args)

    def test_init_with_user_features(self, tmp_path):
        """Should parse colon-separated features from args."""
        pos_df, neg_df = self._make_pos_neg_dfs()
        split_manifest = {
            "split_mode": "chromosome_kfold",
            "chrom_to_fold": {"chr1": 0, "chr2": 0},
            "k_folds": 1,
            "holdout_chromosomes": [],
        }

        args = _make_mock_args(tmp_path)
        args.features = "X_HMER_REF:X_HMER_ALT:MQUAL"

        def mock_read_parquet(path, *a, **kw):
            if "positive" in str(path) or path == args.positive:
                return pos_df
            return neg_df

        with (
            patch("ugbio_srsnv.srsnv_training.pl.read_parquet", side_effect=mock_read_parquet),
            patch("ugbio_srsnv.srsnv_training._extract_stats_from_unified") as mock_stats,
            patch("ugbio_srsnv.srsnv_training._count_bases_in_interval_list", return_value=1000000),
            patch("ugbio_srsnv.srsnv_training._validate_quality_region_filters"),
            patch("ugbio_srsnv.srsnv_training._last_non_downsample_funnel", return_value=80000),
            patch("ugbio_srsnv.srsnv_training._build_new_split_manifest", return_value=split_manifest),
        ):
            pos_stats = {
                "filters": [
                    {"name": "raw", "type": "raw", "funnel": 10000},
                    {"name": "quality", "type": "quality", "funnel": 8000},
                    {"name": "label_filter", "type": "label", "funnel": 5000},
                ]
            }
            neg_stats = {
                "filters": [
                    {"name": "raw", "type": "raw", "funnel": 100000},
                    {"name": "region", "type": "region", "funnel": 80000},
                ]
            }
            mock_stats.return_value = (pos_stats, neg_stats)
            trainer = SRSNVTrainer(args)

        assert trainer.feature_list == ["X_HMER_REF", "X_HMER_ALT", "MQUAL"]


# ──────────────────────── _assign_kfold_folds ──────────────────────


class TestAssignKfoldFolds:
    """Tests for SRSNVTrainer._assign_kfold_folds."""

    def test_assigns_folds_from_manifest(self):
        """Should map CHROM values to fold IDs using split manifest."""
        trainer = object.__new__(SRSNVTrainer)
        trainer.split_manifest = {
            "split_mode": "chromosome_kfold",
            "chrom_to_fold": {"chr1": 0, "chr2": 1, "chr3": 0},
        }
        trainer.data_frame = pl.DataFrame(
            {
                CHROM: pl.Series(["chr1", "chr2", "chr3", "chr1", "chr2"]).cast(pl.Categorical),
                POS: [1, 2, 3, 4, 5],
                LABEL_COL: [True, False, True, False, True],
            }
        )

        trainer._assign_kfold_folds()

        fold_values = trainer.data_frame[FOLD_COL].to_list()
        assert fold_values == [0, 1, 0, 0, 1]
        assert trainer.chrom_to_fold == {"chr1": 0, "chr2": 1, "chr3": 0}


# ──────────────────────── _init_gpu ──────────────────────


class TestInitGpu:
    """Tests for SRSNVTrainer._init_gpu."""

    def test_returns_false_when_disabled(self):
        """Should return False immediately when use_gpu=False."""
        trainer = object.__new__(SRSNVTrainer)
        result = trainer._init_gpu(use_gpu=False)
        assert result is False

    def test_returns_false_on_gpu_error(self):
        """Should fallback to False when GPU test fails."""
        trainer = object.__new__(SRSNVTrainer)
        with patch("ugbio_srsnv.srsnv_training.xgb.XGBClassifier") as mock_xgb:
            mock_xgb.return_value.fit.side_effect = RuntimeError("No CUDA")
            result = trainer._init_gpu(use_gpu=True)
        assert result is False


# ──────────────────────── _determine_x_lut_max ──────────────────────


class TestDetermineXLutMax:
    def _make_trainer(self):
        return object.__new__(SRSNVTrainer)

    def test_fp_cutoff_type(self):
        """mqual_cutoff_type='fp' uses label==0 quantile."""
        trainer = self._make_trainer()
        pd_df = pd.DataFrame({LABEL_COL: [True, True, False, False, False], MQUAL: [50.0, 60.0, 10.0, 20.0, 30.0]})
        result = trainer._determine_x_lut_max(None, pd_df, "fp", 1.0)
        assert result == 30.0

    def test_tp_cutoff_type(self):
        """mqual_cutoff_type='tp' uses label==1 quantile."""
        trainer = self._make_trainer()
        pd_df = pd.DataFrame({LABEL_COL: [True, True, False, False], MQUAL: [50.0, 60.0, 10.0, 20.0]})
        result = trainer._determine_x_lut_max(None, pd_df, "tp", 1.0)
        assert result == 60.0

    def test_mp_with_kde_metadata(self):
        """mqual_cutoff_type='mp' uses KDE truncation if available."""
        trainer = self._make_trainer()
        pd_df = pd.DataFrame({LABEL_COL: [True, False], MQUAL: [50.0, 10.0]})

        class MockEstimator:
            kde_metadata = {"rates": {"false_truncation_idx": 3}}

            def from_grid(self, idx):
                return 42.0

        result = trainer._determine_x_lut_max(MockEstimator(), pd_df, "mp", 1.0)
        assert result == 42.0

    def test_mp_without_truncation(self):
        """mqual_cutoff_type='mp' falls back to quantile when no truncation."""
        trainer = self._make_trainer()
        pd_df = pd.DataFrame({LABEL_COL: [True, False, False], MQUAL: [50.0, 10.0, 20.0]})

        class MockEstimator:
            kde_metadata = {"rates": {"false_truncation_idx": 0}}

        result = trainer._determine_x_lut_max(MockEstimator(), pd_df, "mp", 1.0)
        assert result == 20.0


# ──────────────────────── _create_quality_lookup_table_count ──────────────────────


class TestCreateQualityLookupTableCount:
    def _make_trainer(self):
        trainer = object.__new__(SRSNVTrainer)
        trainer.eps = 1e-6
        trainer.pos_stats = {
            "filters": [{"type": "raw", "funnel": 1000, "pass": 1000}, {"type": "label", "funnel": 1000, "pass": 50}]
        }
        trainer.mean_coverage = 30.0
        trainer.n_bases_in_region = 1000
        trainer.raw_featuremap_size_filtered = 500
        trainer.args = argparse.Namespace(quality_lut_size=None)
        return trainer

    def test_creates_lut_arrays(self):
        """Should create x_lut and y_lut numpy arrays."""
        trainer = self._make_trainer()
        trainer.data_frame = pl.DataFrame(
            {LABEL_COL: [True, True, True, False, False, False], MQUAL: [10.0, 20.0, 30.0, 5.0, 15.0, 25.0]}
        )
        trainer._create_quality_lookup_table_count()
        assert hasattr(trainer, "x_lut")
        assert hasattr(trainer, "y_lut")
        assert len(trainer.x_lut) > 0
        assert len(trainer.x_lut) == len(trainer.y_lut)

    def test_custom_lut_size(self):
        """quality_lut_size overrides default."""
        trainer = self._make_trainer()
        trainer.args.quality_lut_size = 10
        trainer.data_frame = pl.DataFrame({LABEL_COL: [True, True, False, False], MQUAL: [10.0, 20.0, 5.0, 15.0]})
        trainer._create_quality_lookup_table_count()
        assert len(trainer.x_lut) == 10


# ──────────────────────── _create_quality_lookup_table_kde ──────────────────────


class TestCreateQualityLookupTableKde:
    def _make_trainer(self):
        trainer = object.__new__(SRSNVTrainer)
        trainer.eps = 1e-6
        trainer.pos_stats = {
            "filters": [{"type": "raw", "funnel": 1000, "pass": 1000}, {"type": "label", "funnel": 1000, "pass": 50}]
        }
        trainer.neg_stats = {"filters": [{"type": "quality", "funnel": 1000, "pass": 500}]}
        trainer.mean_coverage = 30.0
        trainer.n_bases_in_region = 1000
        trainer.k_folds = 2
        trainer.raw_featuremap_size_filtered = 500
        trainer.args = argparse.Namespace(quality_lut_size=None)
        return trainer

    def test_successful_kde(self):
        """When recalibrate_snvq_kde succeeds, should set x_lut and y_lut."""
        trainer = self._make_trainer()
        trainer.data_frame = pl.DataFrame({LABEL_COL: [True, True, False, False], MQUAL: [10.0, 20.0, 5.0, 15.0]})
        mock_x = np.array([0.0, 10.0, 20.0])
        mock_y = np.array([0.0, 30.0, 60.0])
        with patch("ugbio_srsnv.srsnv_training.recalibrate_snvq_kde", return_value=(None, mock_x, mock_y)):
            trainer._create_quality_lookup_table_kde()
        np.testing.assert_array_equal(trainer.x_lut, mock_x)
        np.testing.assert_array_equal(trainer.y_lut, mock_y)

    def test_kde_failure_falls_back(self):
        """When recalibrate_snvq_kde raises, should fallback to count method."""
        trainer = self._make_trainer()
        trainer.data_frame = pl.DataFrame({LABEL_COL: [True, True, False, False], MQUAL: [10.0, 20.0, 5.0, 15.0]})
        with patch("ugbio_srsnv.srsnv_training.recalibrate_snvq_kde", side_effect=ValueError("KDE failed")):
            trainer._create_quality_lookup_table_kde()
        assert hasattr(trainer, "x_lut")
        assert hasattr(trainer, "y_lut")

    def test_insufficient_data_falls_back(self):
        """When all labels are same, should call the count fallback."""
        trainer = self._make_trainer()
        trainer.data_frame = pl.DataFrame({LABEL_COL: [True, True, True, True], MQUAL: [10.0, 20.0, 30.0, 40.0]})
        with patch.object(trainer, "_create_quality_lookup_table_count") as mock_count:
            trainer._create_quality_lookup_table_kde()
        mock_count.assert_called_once()


# ──────────────────────── _create_quality_lookup_table ──────────────────────


class TestCreateQualityLookupTableDispatcher:
    def _make_trainer(self):
        trainer = object.__new__(SRSNVTrainer)
        trainer.eps = 1e-6
        trainer.pos_stats = {
            "filters": [{"type": "raw", "funnel": 1000, "pass": 1000}, {"type": "label", "funnel": 1000, "pass": 50}]
        }
        trainer.neg_stats = {"filters": [{"type": "quality", "funnel": 1000, "pass": 500}]}
        trainer.mean_coverage = 30.0
        trainer.n_bases_in_region = 1000
        trainer.k_folds = 2
        trainer.raw_featuremap_size_filtered = 500
        trainer.args = argparse.Namespace(quality_lut_size=None)
        trainer.data_frame = pl.DataFrame({LABEL_COL: [True, True, False, False], MQUAL: [10.0, 20.0, 5.0, 15.0]})
        return trainer

    def test_use_kde_true(self):
        """use_kde=True calls the KDE path."""
        trainer = self._make_trainer()
        mock_x = np.array([0.0, 10.0])
        mock_y = np.array([0.0, 50.0])
        with patch("ugbio_srsnv.srsnv_training.recalibrate_snvq_kde", return_value=(None, mock_x, mock_y)):
            trainer._create_quality_lookup_table(use_kde=True)
        np.testing.assert_array_equal(trainer.x_lut, mock_x)

    def test_use_kde_false(self):
        """use_kde=False calls the count path."""
        trainer = self._make_trainer()
        trainer._create_quality_lookup_table(use_kde=False)
        assert hasattr(trainer, "x_lut")


# ──────────────────────── _assign_chrom_val_folds / _assign_read_hash_folds ──────


class TestAssignFoldMethods:
    def test_assign_chrom_val_folds(self):
        """Should assign fold_id based on chromosome role in manifest."""
        trainer = object.__new__(SRSNVTrainer)
        trainer.split_manifest = {
            "split_mode": SPLIT_MODE_SINGLE_MODEL_CHROM_VAL,
            "train_chromosomes": ["chr1", "chr2"],
            "val_chromosomes": ["chr3"],
            "test_chromosomes": ["chr21"],
        }
        trainer.data_frame = pl.DataFrame(
            {CHROM: ["chr1", "chr2", "chr3", "chr21"], POS: [1, 2, 3, 4], LABEL_COL: [True, False, True, False]}
        )
        trainer._assign_chrom_val_folds()

        assert trainer.k_folds == 1
        fold_col = trainer.data_frame[FOLD_COL].to_list()
        assert fold_col[0] == 0  # chr1 -> train
        assert fold_col[1] == 0  # chr2 -> train
        assert fold_col[2] == 1  # chr3 -> val
        assert fold_col[3] is None  # chr21 -> test

    def test_assign_read_hash_folds(self):
        """Should assign fold_id based on read hash."""
        trainer = object.__new__(SRSNVTrainer)
        trainer.split_hash_key = "RN"
        trainer.split_manifest = {
            "split_mode": SPLIT_MODE_SINGLE_MODEL_READ_HASH,
            "train_chromosomes": ["chr1", "chr2"],
            "test_chromosomes": ["chr21"],
            "val_fraction": 0.2,
            "random_seed": 42,
            "hash_key": "RN",
        }
        trainer.data_frame = pl.DataFrame(
            {
                CHROM: ["chr1", "chr1", "chr1", "chr21"],
                POS: [1, 2, 3, 4],
                "RN": ["read_a", "read_b", "read_c", "read_d"],
                LABEL_COL: [True, False, True, False],
            }
        )
        trainer._assign_read_hash_folds()

        assert trainer.k_folds == 1
        fold_col = trainer.data_frame[FOLD_COL].to_list()
        assert fold_col[3] is None  # chr21 -> test
        assert all(val in (0, 1, None) for val in fold_col)

    def test_assign_read_hash_missing_column_raises(self):
        """Should raise ValueError if hash key column is missing."""
        trainer = object.__new__(SRSNVTrainer)
        trainer.split_hash_key = "RN"
        trainer.split_manifest = {"split_mode": SPLIT_MODE_SINGLE_MODEL_READ_HASH, "test_chromosomes": ["chr21"]}
        trainer.data_frame = pl.DataFrame({CHROM: ["chr1"], POS: [1], LABEL_COL: [True]})
        with pytest.raises(ValueError, match="hash key column"):
            trainer._assign_read_hash_folds()


# ──────────────────────── _init_split_manifest ──────────────────────


class TestInitSplitManifestMethod:
    def test_loads_existing_manifest(self, tmp_path):
        """Should load manifest from file."""
        manifest = {"split_mode": "k_fold", "chrom_to_fold": {"chr1": 0}, "k_folds": 2}
        manifest_file = tmp_path / "manifest.json"
        manifest_file.write_text(json.dumps(manifest))

        trainer = object.__new__(SRSNVTrainer)
        trainer.seed = 42
        args = argparse.Namespace(
            training_regions=str(tmp_path / "regions.bed"),
            k_folds=2,
            split_manifest_in=str(manifest_file),
            split_manifest_out=None,
            holdout_chromosomes=None,
            val_chromosomes=None,
            single_model_split=False,
            val_fraction=0.1,
            split_hash_key="RN",
        )
        with patch("ugbio_srsnv.srsnv_training.validate_manifest_against_regions"):
            trainer._init_split_manifest(args)
        assert trainer.split_manifest == manifest

    def test_saves_manifest(self, tmp_path):
        """Should save manifest when split_manifest_out given."""
        trainer = object.__new__(SRSNVTrainer)
        trainer.seed = 42
        manifest_out = str(tmp_path / "out.json")
        args = argparse.Namespace(
            training_regions=str(tmp_path / "regions.bed"),
            k_folds=2,
            split_manifest_in=None,
            split_manifest_out=manifest_out,
            holdout_chromosomes=None,
            val_chromosomes=None,
            single_model_split=False,
            val_fraction=0.1,
            split_hash_key="RN",
        )
        mock_manifest = {"split_mode": "k_fold", "chrom_to_fold": {"chr1": 0}}
        with (
            patch("ugbio_srsnv.srsnv_training._build_new_split_manifest", return_value=mock_manifest),
            patch("ugbio_srsnv.srsnv_training.save_split_manifest") as mock_save,
        ):
            trainer._init_split_manifest(args)
        mock_save.assert_called_once_with(mock_manifest, manifest_out)


# ──────────────────────── train() integration ──────────────────────


class TestTrainIntegration:
    def test_train_with_synthetic_data(self):
        """Full train() with tiny synthetic data, 1 fold."""
        trainer = object.__new__(SRSNVTrainer)
        trainer.k_folds = 1
        trainer.downcast_float = False
        trainer.args = argparse.Namespace(verbose=False, use_kde_smoothing=False, quality_lut_size=5)
        trainer.pos_stats = {
            "filters": [{"type": "raw", "funnel": 1000, "pass": 1000}, {"type": "label", "funnel": 1000, "pass": 50}]
        }
        trainer.neg_stats = {"filters": [{"type": "quality", "funnel": 1000, "pass": 500}]}
        trainer.mean_coverage = 30.0
        trainer.n_bases_in_region = 1000
        trainer.raw_featuremap_size_filtered = 500
        trainer.max_qual = 60.0
        trainer.eps = 1e-6
        trainer.feature_list = None

        rng = np.random.default_rng(42)
        n_samples = 40
        trainer.data_frame = pl.DataFrame(
            {
                CHROM: ["chr1"] * n_samples,
                POS: list(range(n_samples)),
                "feat1": rng.normal(0, 1, n_samples).tolist(),
                "feat2": rng.normal(0, 1, n_samples).tolist(),
                LABEL_COL: ([True] * 20) + ([False] * 20),
                FOLD_COL: [0] * n_samples,
            }
        )
        trainer.models = [xgb.XGBClassifier(n_estimators=5, max_depth=2, eval_metric="auc", early_stopping_rounds=3)]

        trainer.train()

        assert hasattr(trainer, "x_lut")
        assert hasattr(trainer, "y_lut")
        assert MQUAL in trainer.data_frame.columns
        assert SNVQ in trainer.data_frame.columns
        assert PROB_ORIG in trainer.data_frame.columns
