from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest
from ugbio_featuremap.featuremap_to_dataframe import vcf_to_parquet
from ugbio_featuremap.filter_dataframe import filter_parquet, validate_filter_config

SAMPLE = (
    Path(__file__).parent.parent / "resources" / "416119-L7402-Z0296-CATCTATCAGGCGAT.snvfind_out_f2_test_sample.vcf.gz"
)


def _parquet_from_sample(tmpdir: Path) -> Path:
    out = tmpdir / "sample.parquet"
    vcf_to_parquet(str(SAMPLE), str(out), drop_format={"GT"})
    return out


def test_filter_pipeline(tmp_path: Path) -> None:
    parquet_in = _parquet_from_sample(tmp_path)

    cfg = {
        "filters": [
            {"name": "bcsq_ge_40", "field": "BCSQ", "op": "gt", "value": 40, "type": "quality"},
            {"name": "ref_eq_alt", "field": "REF", "op": "eq", "value_field": "ALT", "type": "label"},
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
    assert stats["filters"][0]["type"] is None

    # Check filter types are preserved
    assert stats["filters"][1]["type"] == "quality"
    assert stats["filters"][2]["type"] == "label"

    # downsample size honoured?
    assert stats["filters"][-1]["rows"] <= cfg["downsample"]["size"]

    # Check filtered output
    featuremap_dataframe = pl.read_parquet(out_pq)
    # BCSQ > 40 and REF == ALT should both hold
    assert (featuremap_dataframe["BCSQ"] > 40).all()
    assert (featuremap_dataframe["REF"] == featuremap_dataframe["ALT"]).all()

    # Check full output with filter columns
    full_df = pl.read_parquet(out_pq_full)
    assert "__filter_bcsq_ge_40" in full_df.columns
    assert "__filter_ref_eq_alt" in full_df.columns
    assert "__filter_final" in full_df.columns

    # Check combinations exist
    assert len(stats["combinations"]) > 0
    # The "00" pattern should only exist if there are rows that fail all filters


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
