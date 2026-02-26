import json
from pathlib import Path
from typing import Any

import pytest
from ugbio_srsnv.srsnv_training import (
    _extract_stats_from_unified,
    _parse_model_params,
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


def test_extract_stats_from_unified_missing_section():
    """Test _extract_stats_from_unified with missing required sections."""
    import tempfile

    # Test missing filtering_stats_random_sample
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"filtering_stats_full_output": {"filters": {}}}, f)
        temp_path = f.name

    with pytest.raises(ValueError, match="missing 'filtering_stats_random_sample' section"):
        _extract_stats_from_unified(temp_path)

    # Clean up
    Path(temp_path).unlink()

    # Test missing filtering_stats_full_output
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"filtering_stats_random_sample": {"filters": {}}}, f)
        temp_path = f.name

    with pytest.raises(ValueError, match="missing 'filtering_stats_full_output' section"):
        _extract_stats_from_unified(temp_path)

    # Clean up
    Path(temp_path).unlink()


def test_downsample_segments_added_to_metadata(tmp_path: Path, resources_dir: Path) -> None:
    """Test that downsample segments are added to positive and negative stats in metadata."""
    import argparse

    from ugbio_srsnv.srsnv_training import SRSNVTrainer

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
        transform_mode="logit",
        mqual_cutoff_quantile=0.99,
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
