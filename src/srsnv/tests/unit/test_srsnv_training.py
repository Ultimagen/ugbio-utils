import json
from pathlib import Path
from typing import Any

import pytest
from ugbio_srsnv.srsnv_training import (
    _convert_unified_stats_to_legacy_format,
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
    stats_file = resources_dir / "416119_L7402.test.unified_stats_new_format.json"
    pos_stats, neg_stats, raw_stats = _extract_stats_from_unified(stats_file)

    # Verify structure
    assert "filters" in pos_stats
    assert "filters" in neg_stats
    assert "filters" in raw_stats

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


def test_convert_unified_stats_to_legacy_format():
    """Test _convert_unified_stats_to_legacy_format function."""
    unified_stats = {
        "filtering_stats_random_sample": {
            "filters": {
                "raw": {"funnel": 1000, "type": "raw"},
                "filter1": {"funnel": 800, "type": "quality", "field": "MAPQ", "op": "ge", "value": 60},
            }
        }
    }

    result = _convert_unified_stats_to_legacy_format(unified_stats, "filtering_stats_random_sample", "filters")

    assert "filters" in result
    filters_list = result["filters"]
    assert len(filters_list) == 2

    # Check raw filter is first
    assert filters_list[0]["name"] == "raw"
    assert filters_list[0]["type"] == "raw"
    assert filters_list[0]["rows"] == 1000

    # Check other filter
    filter1 = next((f for f in filters_list if f["name"] == "filter1"), None)
    assert filter1 is not None
    assert filter1["type"] == "quality"
    assert filter1["field"] == "MAPQ"
    assert filter1["op"] == "ge"
    assert filter1["value"] == 60
    assert filter1["rows"] == 800


def test_convert_unified_stats_to_legacy_format_missing_section():
    """Test _convert_unified_stats_to_legacy_format with missing sections."""
    unified_stats = {}

    with pytest.raises(ValueError, match="Section 'missing_section' not found"):
        _convert_unified_stats_to_legacy_format(unified_stats, "missing_section", "filters")

    unified_stats = {"section1": {}}
    with pytest.raises(ValueError, match="Filter key 'missing_filter' not found"):
        _convert_unified_stats_to_legacy_format(unified_stats, "section1", "missing_filter")
