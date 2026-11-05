"""Coverage-oriented unit tests for ugbio_srsnv.srsnv_report (variant-type labeling and the CLI entry point)."""

import logging
import runpy
import sys
from unittest.mock import patch

import pandas as pd
import pytest
from ugbio_srsnv import srsnv_report


def test_compute_variant_type_without_indel_class_defaults_to_hmer():
    """Backward compat: older featuremaps have X_IC but no X_INDEL_CLASS -> every indel is an hmer indel."""
    fm = pd.DataFrame({"X_IC": ["ins", "DEL", None, ".", "snv"]}, index=[10, 11, 12, 13, 14])
    vt = srsnv_report.compute_variant_type_column(fm)
    assert list(vt) == ["hmer_indel", "hmer_indel", "snv", "snv", "snv"]
    assert list(vt.index) == [10, 11, 12, 13, 14]


def test_compute_variant_type_with_indel_class_and_snv_only():
    fm = pd.DataFrame({"X_IC": ["ins", "del", "ins", None], "X_INDEL_CLASS": ["hmer", "NON_HMER", ".", "non_hmer"]})
    assert list(srsnv_report.compute_variant_type_column(fm)) == ["hmer_indel", "non_hmer_indel", "hmer_indel", "snv"]
    assert set(srsnv_report.compute_variant_type_column(pd.DataFrame({"REF": ["A", "C"]}))) == {"snv"}


def test_cli_parses_all_arguments(monkeypatch):
    argv = [
        "srsnv_report",
        "--featuremap-df",
        "fm.parquet",
        "--srsnv-metadata",
        "meta.json",
        "--report-path",
        "out",
        "--basename",
        "s1",
        "--models-prefix",
        "m.model_fold_",
        "--random-seed",
        "7",
        "--verbose",
        "--use-gpu-for-shap",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = srsnv_report._cli()
    assert args.featuremap_df == "fm.parquet"
    assert args.random_seed == 7
    assert args.models_prefix == "m.model_fold_"
    assert args.verbose is True and args.use_gpu_for_shap is True


def test_cli_requires_mandatory_arguments(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["srsnv_report", "--featuremap-df", "fm.parquet"])
    with pytest.raises(SystemExit) as exc:
        srsnv_report._cli()
    assert exc.value.code == 2


def test_main_forwards_arguments_and_sets_debug(monkeypatch):
    argv = ["srsnv_report", "--featuremap-df", "fm.parquet", "--srsnv-metadata", "meta.json", "--report-path", "out"]
    monkeypatch.setattr(sys, "argv", [*argv, "--verbose"])
    old_level = srsnv_report.logger.level
    try:
        with patch.object(srsnv_report, "prepare_report") as prep:
            srsnv_report.main()
        assert srsnv_report.logger.level == logging.DEBUG
    finally:
        srsnv_report.logger.setLevel(old_level)
    prep.assert_called_once_with(
        featuremap_df="fm.parquet",
        srsnv_metadata="meta.json",
        report_path="out",
        basename="",
        models_prefix=None,
        random_seed=None,
        use_gpu_for_shap=False,
    )


def test_module_main_guard_runs_cli(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["srsnv_report", "--help"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("ugbio_srsnv.srsnv_report", run_name="__main__", alter_sys=False)
    assert exc.value.code == 0
