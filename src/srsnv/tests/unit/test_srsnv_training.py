import argparse
import gzip
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from ugbio_srsnv.srsnv_training import (
    FILTERS_FULL_OUTPUT,
    FILTERS_RANDOM_SAMPLE,
    SRSNVTrainer,
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
        content = (
            "@HD\tVN:1.6\n" "@SQ\tSN:chr1\tLN:1000\n" "# comment line\n" "chr1\t10\t20\t+\tinterval1\n"  # 11 bases
        )
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
