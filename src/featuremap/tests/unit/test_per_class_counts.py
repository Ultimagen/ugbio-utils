"""Unit test for per-class pre-downsample row counts (featuremap_to_dataframe._write_per_class_counts)."""

import json

import polars as pl
from ugbio_featuremap.featuremap_to_dataframe import _write_per_class_counts


def _part(path, x_ic, x_cls):
    pl.DataFrame({"X_IC": x_ic, "X_INDEL_CLASS": x_cls}).write_parquet(path)
    return str(path)


def test_counts_three_classes(tmp_path):
    p1 = _part(
        tmp_path / "p1.parquet", [None, None, None, "ins", "ins", "del"], [None, None, None, "hmer", "hmer", "non_hmer"]
    )
    p2 = _part(tmp_path / "p2.parquet", ["del", "ins"], ["hmer", "non_hmer"])
    out = tmp_path / "c.json"
    counts = _write_per_class_counts([p1, p2], str(out))
    assert counts == {"snv": 3, "hmer_indel": 3, "non_hmer_indel": 2, "total": 8}
    assert json.loads(out.read_text()) == counts


def test_counts_snv_only_no_x_ic(tmp_path):
    p = tmp_path / "p.parquet"
    pl.DataFrame({"MQUAL": [1, 2, 3]}).write_parquet(p)
    counts = _write_per_class_counts([str(p)], str(tmp_path / "c.json"))
    assert counts == {"snv": 3, "hmer_indel": 0, "non_hmer_indel": 0, "total": 3}


def test_counts_no_indel_class_treats_indels_as_hmer(tmp_path):
    p = tmp_path / "p.parquet"
    pl.DataFrame({"X_IC": [None, "ins", "del"]}).write_parquet(p)  # no X_INDEL_CLASS column
    counts = _write_per_class_counts([str(p)], str(tmp_path / "c.json"))
    assert counts == {"snv": 1, "hmer_indel": 2, "non_hmer_indel": 0, "total": 3}
