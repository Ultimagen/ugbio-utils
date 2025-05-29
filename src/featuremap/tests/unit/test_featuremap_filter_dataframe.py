from __future__ import annotations

import json
from pathlib import Path

import polars as pl
from ugbio_featuremap.featuremap_to_dataframe import vcf_to_parquet
from ugbio_featuremap.filter_dataframe import filter_parquet

SAMPLE = Path(__file__).parent / "resources" / "416119-L7402-Z0296-CATCTATCAGGCGAT.snvfind_out_f2_test_sample.vcf.gz"


def _parquet_from_sample(tmpdir: Path) -> Path:
    out = tmpdir / "sample.parquet"
    vcf_to_parquet(str(SAMPLE), str(out), drop_format={"GT"})
    return out


def test_filter_pipeline(tmp_path: Path) -> None:
    parquet_in = _parquet_from_sample(tmp_path)

    cfg = {
        "filters": [
            {"name": "bcsq_ge_40", "field": "BCSQ", "op": "gt", "value": 40},
            {"name": "ref_eq_alt", "field": "REF", "op": "eq", "value_field": "ALT"},
        ],
        "downsample": {"method": "random", "size": 100, "seed": 42},
    }
    cfg_path = tmp_path / "cfg.json"
    json.dump(cfg, cfg_path.open("w"))

    out_pq = tmp_path / "f.parquet"
    stats_json = tmp_path / "stats.json"

    filter_parquet(str(parquet_in), str(out_pq), str(cfg_path), str(stats_json))

    # -------- assertions -------------------------------------------------
    assert out_pq.exists() and stats_json.exists()

    stats = json.load(stats_json.open())
    assert stats["filters"][0]["name"] == "raw"
    # downsample size honoured?
    assert stats["filters"][-1]["rows"] <= cfg["downsample"]["size"]

    featuremap_dataframe = pl.read_parquet(out_pq)
    # VAF > 0.05 and REF == ALT should both hold
    assert (featuremap_dataframe["VAF"] > 0.05).all()
    assert (featuremap_dataframe["REF"] == featuremap_dataframe["ALT"]).all()
