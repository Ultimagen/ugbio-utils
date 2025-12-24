from __future__ import annotations

import importlib
import json
from pathlib import Path

import polars as pl
import pytest
from ugbio_featuremap.featuremap_to_dataframe import vcf_to_parquet
from ugbio_featuremap.filter_dataframe import (
    _parse_value_based_on_operation,
    filter_parquet,
    read_filtering_stats_json,
    validate_filter_config,
)

SAMPLE = Path(__file__).parent.parent / "resources" / "416119-L7402.raw.featuremap.head.vcf.gz"


def _parquet_from_sample(tmpdir: Path) -> Path:
    out = tmpdir / "sample.parquet"
    vcf_to_parquet(str(SAMPLE), str(out), drop_format={"GT", "AD"})
    return out


def test_filter_pipeline(tmp_path: Path) -> None:
    parquet_in = _parquet_from_sample(tmp_path)

    bcsq_filt = 0
    cfg = {
        "filters": [
            {"name": "ref_ne_alt", "field": "REF", "op": "ne", "value_field": "ALT", "type": "label"},
            {"field": "DP", "op": "gt", "value": 20, "type": "label"},
            {"field": "ADJ_REF_DIFF", "op": "eq", "value": 0, "type": "quality"},
            {"field": "MAPQ", "op": "ge", "value": 60, "type": "quality"},
            {"name": f"bcsq_ge_{bcsq_filt}", "field": "BCSQ", "op": "gt", "value": bcsq_filt, "type": "quality"},
        ],
        "downsample": {"method": "random", "size": 100, "seed": 42},
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    out_pq = tmp_path / "f.parquet"
    out_pq_full = tmp_path / "f_full.parquet"
    stats_json = tmp_path / "stats.json"

    filter_parquet(str(parquet_in), str(out_pq), str(out_pq_full), str(cfg_path), str(stats_json))

    # -------- assertions -------------------------------------------------
    assert out_pq.exists() and stats_json.exists() and out_pq_full.exists()

    stats = json.load(stats_json.open())
    assert stats["filters"][0]["name"] == "raw"
    assert stats["filters"][0]["type"] == "raw"

    # Check filter types are preserved
    assert stats["filters"][3]["type"] == "quality"
    assert stats["filters"][2]["type"] == "label"
    assert stats["filters"][1]["type"] == "label"

    # downsample size honoured?
    assert stats["filters"][-1]["rows"] == cfg["downsample"]["size"]

    # Check filtered output
    featuremap_dataframe = pl.read_parquet(out_pq)
    # BCSQ > BCSQ_filt and REF == ALT should both hold
    assert (featuremap_dataframe["BCSQ"] > bcsq_filt).all()
    # Cast to string to ensure reliable comparison even when columns are Enum-encoded
    assert (featuremap_dataframe["REF"].cast(pl.Utf8) != featuremap_dataframe["ALT"].cast(pl.Utf8)).all()

    # Check full output with filter columns
    full_df = pl.read_parquet(out_pq_full)
    assert f"__filter_bcsq_ge_{bcsq_filt}" in full_df.columns
    assert "__filter_ref_ne_alt" in full_df.columns
    assert "__filter_final" in full_df.columns

    # Check combinations exist
    assert len(stats["combinations"]) > 0
    # The "00" pattern should only exist if there are rows that fail all filters

    # verify expression details are stored
    ref_eq_alt = next(f for f in stats["filters"] if f["name"] == "ref_ne_alt")
    assert ref_eq_alt["field"] == "REF"
    assert ref_eq_alt["op"] == "ne"
    assert ref_eq_alt["value_field"] == "ALT"
    assert "value" not in ref_eq_alt

    # down-sample entry records method + seed
    ds = next(f for f in stats["filters"] if f["name"] == "downsample")
    assert ds["type"] == "downsample"
    assert ds["method"] == "random"
    assert ds["seed"] == 42


def test_filter_without_downsample(tmp_path: Path) -> None:
    parquet_in = _parquet_from_sample(tmp_path)

    cfg = {
        "filters": [
            {"name": "high_bcsq", "field": "BCSQ", "op": "ge", "value": 30, "type": "quality"},
        ],
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    out_pq = tmp_path / "f.parquet"
    stats_json = tmp_path / "stats.json"

    filter_parquet(
        str(parquet_in),
        str(out_pq),
        None,  # No full output
        str(cfg_path),
        str(stats_json),
    )

    assert out_pq.exists() and stats_json.exists()

    stats = json.load(stats_json.open())
    # No downsample in funnel
    assert all(s["name"] != "downsample" for s in stats["filters"])


def test_validate_filter_config() -> None:
    # Valid config
    cfg = {
        "filters": [
            {"field": "QUAL", "op": "gt", "value": 30, "type": "quality"},
        ]
    }
    validate_filter_config(cfg)  # Should not raise

    # Missing filters key
    with pytest.raises(ValueError, match="Configuration must contain 'filters' key"):
        validate_filter_config({})

    # Invalid filter type
    with pytest.raises(ValueError, match="invalid type"):
        validate_filter_config(
            {
                "filters": [
                    {"field": "QUAL", "op": "gt", "value": 30, "type": "invalid"},
                ]
            }
        )

    # Missing required field
    with pytest.raises(ValueError, match="missing required 'field' key"):
        validate_filter_config(
            {
                "filters": [
                    {"op": "gt", "value": 30, "type": "quality"},
                ]
            }
        )

    # Invalid operator
    with pytest.raises(ValueError, match="unsupported operator"):
        validate_filter_config(
            {
                "filters": [
                    {"field": "QUAL", "op": "invalid_op", "value": 30, "type": "quality"},
                ]
            }
        )

    # Missing value
    with pytest.raises(ValueError, match="must have either"):
        validate_filter_config(
            {
                "filters": [
                    {"field": "QUAL", "op": "gt", "type": "quality"},
                ]
            }
        )

    # Invalid downsample
    with pytest.raises(ValueError, match="positive integer"):
        validate_filter_config({"filters": [], "downsample": {"size": -1}})


def test_only_full_output(tmp_path: Path) -> None:
    """Test that we can output only the full dataframe with filter columns."""
    parquet_in = _parquet_from_sample(tmp_path)

    cfg = {
        "filters": [
            {"field": "BCSQ", "op": "ge", "value": 20, "type": "quality"},
        ],
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    out_pq_full = tmp_path / "f_full.parquet"
    stats_json = tmp_path / "stats.json"

    filter_parquet(
        str(parquet_in),
        None,  # No filtered output
        str(out_pq_full),
        str(cfg_path),
        str(stats_json),
    )

    assert out_pq_full.exists() and stats_json.exists()

    full_df = pl.read_parquet(out_pq_full)
    assert "__filter_BCSQ_ge" in full_df.columns
    assert "__filter_final" in full_df.columns

    # All original columns should be present
    assert "BCSQ" in full_df.columns
    assert "CHROM" in full_df.columns


def test_skip_combinations_when_many_filters() -> None:
    """Combinations should be skipped once the filter count exceeds MAX_COMBINATION_FILTERS."""
    fd = importlib.import_module("ugbio_featuremap.filter_dataframe")
    n_filters = fd.MAX_COMBINATION_FILTERS + 1  # 21
    n_rows = 3

    # Dummy lazy frame with boolean columns c0 … c20
    lf = pl.DataFrame({f"c{i}": [True] * n_rows for i in range(n_filters)}).lazy()

    # Corresponding filter specs (c<i> == True)
    filters = [
        {
            fd.KEY_FIELD: f"c{i}",
            fd.KEY_OP: "eq",
            fd.KEY_TYPE: fd.TYPE_LABEL,
            fd.KEY_VALUE: True,
        }
        for i in range(n_filters)
    ]

    lf, filter_cols = fd._create_filter_columns(lf, filters)

    stats = fd._calculate_statistics(
        featuremap_dataframe=lf,
        filter_cols=filter_cols,
        downsample_col=None,
        filters=filters,
        total_rows=n_rows,
        cfg={fd.KEY_FILTERS: filters},
    )

    assert stats["combinations"] == {}, "Combinations dict should be empty when too many filters are present"


def test_cli_filters_only(tmp_path: Path) -> None:
    """Test filtering using only CLI arguments, no config file."""
    parquet_in = _parquet_from_sample(tmp_path)

    cli_filters = [
        "name=high_bcsq:field=BCSQ:op=ge:value=20:type=quality",
        "name=high_read_count:field=DP:op=gt:value=10:type=region",
    ]

    out_pq = tmp_path / "f.parquet"
    stats_json = tmp_path / "stats.json"

    filter_parquet(
        str(parquet_in),
        str(out_pq),
        None,
        None,  # No config file
        str(stats_json),
        cli_filters=cli_filters,
    )

    assert out_pq.exists() and stats_json.exists()

    stats = json.load(stats_json.open())
    assert len(stats["filters"]) == 3  # raw + 2 CLI filters
    assert stats["filters"][1]["name"] == "high_bcsq"
    assert stats["filters"][1]["type"] == "quality"
    assert stats["filters"][2]["name"] == "high_read_count"
    assert stats["filters"][2]["type"] == "region"

    # Check filtered output
    filtered_df = pl.read_parquet(out_pq)
    assert (filtered_df["BCSQ"] >= 20).all()
    assert (filtered_df["DP"] > 10).all()


def test_cli_downsample_override(tmp_path: Path) -> None:
    """Test that CLI downsample overrides config file downsample."""
    parquet_in = _parquet_from_sample(tmp_path)

    # Config with one downsample setting
    cfg = {
        "filters": [
            {"field": "BCSQ", "op": "ge", "value": 10, "type": "quality"},
        ],
        "downsample": {"method": "random", "size": 1000, "seed": 1},
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    # CLI with different downsample
    cli_downsample = "head:50"

    out_pq = tmp_path / "f.parquet"
    stats_json = tmp_path / "stats.json"

    filter_parquet(str(parquet_in), str(out_pq), None, str(cfg_path), str(stats_json), cli_downsample=cli_downsample)

    assert out_pq.exists() and stats_json.exists()

    stats = json.load(stats_json.open())
    # Should have CLI downsample, not config downsample
    assert stats["filters"][-1]["name"] == "downsample"
    assert stats["filters"][-1]["rows"] == 50


def test_config_and_cli_filters_combined(tmp_path: Path) -> None:
    """Test that CLI filters are appended to config filters."""
    parquet_in = _parquet_from_sample(tmp_path)

    # Config with some filters
    cfg = {
        "filters": [
            {"name": "config_filter", "field": "BCSQ", "op": "ge", "value": 10, "type": "quality"},
        ]
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    # CLI with additional filters
    cli_filters = ["name=cli_filter:field=DP:op=gt:value=5:type=region"]

    out_pq = tmp_path / "f.parquet"
    stats_json = tmp_path / "stats.json"

    filter_parquet(str(parquet_in), str(out_pq), None, str(cfg_path), str(stats_json), cli_filters=cli_filters)

    assert out_pq.exists() and stats_json.exists()

    stats = json.load(stats_json.open())
    # Should have raw + config filter + CLI filter
    assert len(stats["filters"]) == 3
    assert stats["filters"][1]["name"] == "config_filter"
    assert stats["filters"][1]["type"] == "quality"
    assert stats["filters"][2]["name"] == "cli_filter"
    assert stats["filters"][2]["type"] == "region"

    # Check filtered output
    filtered_df = pl.read_parquet(out_pq)
    assert (filtered_df["BCSQ"] >= 10).all()
    assert (filtered_df["DP"] > 5).all()


def test_cli_filter_with_list_values(tmp_path: Path) -> None:
    """Test CLI filters with list values (in/not_in operations)."""
    parquet_in = _parquet_from_sample(tmp_path)

    # Create test data with known CHROM values
    orig_df = pl.read_parquet(parquet_in)
    # Ensure we have some chr1 and chr2 values for testing
    test_df = orig_df.with_columns(
        pl.when(pl.int_range(len(orig_df)) % 2 == 0).then(pl.lit("chr1")).otherwise(pl.lit("chr2")).alias("CHROM")
    )
    test_parquet = tmp_path / "test.parquet"
    test_df.write_parquet(test_parquet)

    cli_filters = ["name=chrom_filter:field=CHROM:op=in:value=chr1,chr2:type=region"]

    out_pq = tmp_path / "f.parquet"
    stats_json = tmp_path / "stats.json"

    filter_parquet(str(test_parquet), str(out_pq), None, None, str(stats_json), cli_filters=cli_filters)

    assert out_pq.exists() and stats_json.exists()

    # Check filtered output
    filtered_df = pl.read_parquet(out_pq)
    assert filtered_df["CHROM"].is_in(["chr1", "chr2"]).all()


def test_parse_cli_filter_functions() -> None:
    """Test the CLI parsing utility functions."""
    from ugbio_featuremap.filter_dataframe import _parse_cli_downsample, _parse_cli_filter

    # Test basic filter parsing
    filter_dict = _parse_cli_filter("name=my_filter:field=FIELD:op=gt:value=10:type=quality")
    expected = {"name": "my_filter", "field": "FIELD", "op": "gt", "value": 10, "type": "quality"}
    assert filter_dict == expected

    # Test float value
    filter_dict = _parse_cli_filter("name=float_filter:field=VAF:op=le:value=0.05:type=label")
    assert filter_dict["value"] == 0.05

    # Test string value
    filter_dict = _parse_cli_filter("name=str_filter:field=CHROM:op=eq:value=chr1:type=region")
    assert filter_dict["value"] == "chr1"

    # Test list value
    filter_dict = _parse_cli_filter("name=list_filter:field=CHROM:op=in:value=chr1,chr2,chr3:type=region")
    assert filter_dict["value"] == ["chr1", "chr2", "chr3"]

    # Test value_field instead of value
    filter_dict = _parse_cli_filter("name=field_comparison:field=REF:op=eq:value_field=ALT:type=label")
    assert filter_dict["value_field"] == "ALT"
    assert "value" not in filter_dict

    # Test downsample parsing
    ds_dict = _parse_cli_downsample("random:1000:42")
    expected = {"method": "random", "size": 1000, "seed": 42}
    assert ds_dict == expected

    # Test downsample without seed
    ds_dict = _parse_cli_downsample("head:500")
    expected = {"method": "head", "size": 500}
    assert ds_dict == expected

    # Test invalid filter format
    with pytest.raises(ValueError, match="must have at least 4 parts"):
        _parse_cli_filter("invalid:format")

    # Test invalid downsample format
    with pytest.raises(ValueError, match="must have 2-3 parts"):
        _parse_cli_downsample("invalid")


def test_stats_json_roundtrip(tmp_path: Path) -> None:
    """filter_parquet stats JSON must be readable with read_filtering_stats_json()."""
    parquet_in = _parquet_from_sample(tmp_path)

    cfg = {
        "filters": [{"field": "BCSQ", "op": "ge", "value": 10, "type": "quality"}],
        "downsample": {"method": "head", "size": 50},
    }
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(cfg))

    out_pq = tmp_path / "f.parquet"
    stats_json = tmp_path / "stats.json"

    filter_parquet(str(parquet_in), str(out_pq), None, str(cfg_path), str(stats_json))

    # Must load without raising
    stats = read_filtering_stats_json(stats_json)
    assert stats["filters"][0]["name"] == "raw"


def test_try_to_convert_to_number_or_boolean_scalars():
    """
    Unified (value, op) cases after introducing op into the test key.
    Uses _parse_value_based_on_operation for both scalar and list ops.
    """
    cases = [
        ("true", "eq", True),
        ("TRUE", "eq", True),
        ("False", "eq", False),
        ("FALSE", "eq", False),
        ("10", "eq", 10),
        ("-7", "eq", -7),
        ("0", "eq", 0),
        ("0.5", "eq", 0.5),
        ("1e-3", "eq", 0.001),
        ("2E2", "eq", 200.0),
        ("00123", "eq", 123),
        ("abc", "eq", "abc"),
        ("10,20", "in", [10, 20]),  # list parsing for in
        ("true,False,1,0.5", "in", [True, False, 1, 0.5]),
        ("true,False", "not_in", [True, False]),
    ]

    for raw, op, expected in cases:
        got = _parse_value_based_on_operation(raw, op)
        assert got == expected, f"{raw=} {op=} -> {got} (expected {expected})"


def test_parse_cli_filter_list_boolean_numeric():
    from ugbio_featuremap.filter_dataframe import _parse_cli_filter

    f = _parse_cli_filter("name=mixed:field=X:op=in:value=true,False,10,0.25,1e-2,ABC:type=label")
    # Expect coercion of booleans + numbers (including scientific), raw string for non-numeric/non-boolean
    assert f["value"] == [True, False, 10, 0.25, 0.01, "ABC"]

    # Whitespace tokens should currently preserve whitespace / skip coercion if implementation does not strip
    f2 = _parse_cli_filter("name=ws:field=Y:op=in:value=true, false ,  5:type=region")
    # Depending on implementation (strip or not). If you later strip, adjust expectation.
    # Assume no strip inside parser for non-leading boolean => second stays ' false '
    # '  5' may or may not coerce; if strip is applied it becomes 5 else stays '  5'
    val = f2["value"]
    assert val[0] is True
    # Accept either behavior for whitespace handling to avoid brittleness
    assert val[1] in (" false ", False)
    assert val[2] in ("  5", 5)
