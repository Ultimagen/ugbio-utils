import numpy as np
import pandas as pd
import pytest
from ugbio_comparison.reports.concordance_report_utils import ReportUtils

filter_by_category = ReportUtils._ReportUtils__filter_by_category


@pytest.fixture
def variants():
    hmer_lengths = list(range(25))
    return pd.DataFrame(
        {
            "indel": [False] + [True] * len(hmer_lengths),
            "hmer_length": [0, *hmer_lengths],
            "indel_length": [0, 2] + [1] * (len(hmer_lengths) - 1),
        }
    )


@pytest.mark.parametrize(
    ("category", "expected_hmer_lengths"),
    [
        ("non-hmer Indel", [0]),
        ("hmer Indel <=4", [1, 2, 3, 4]),
        ("hmer Indel >4,<=8", [5, 6, 7, 8]),
        ("hmer Indel >8,<=10", [9, 10]),
        ("hmer Indel >10,<=12", [11, 12]),
        ("hmer Indel >12,<=14", [13, 14]),
        ("hmer Indel >15,<=19", [15, 16, 17, 18, 19]),
        ("hmer Indel >=20", [20, 21, 22, 23, 24]),
        ("non-hmer Indel + hmer Indel <=12", list(range(13))),
        ("hmer Indel 1", [1]),
        ("hmer Indel 9", [9]),
    ],
)
def test_filter_by_category(variants, category, expected_hmer_lengths):
    assert filter_by_category(variants, category)["hmer_length"].tolist() == expected_hmer_lengths


def test_filter_by_category_snp_and_indel(variants):
    assert len(filter_by_category(variants, "SNP")) == 1
    assert len(filter_by_category(variants, "Indel")) == 25


def test_filter_by_category_unknown(variants):
    with pytest.raises(RuntimeError, match="No such category"):
        filter_by_category(variants, "hmer Indel 10")


def test_plot_performance_keeps_categories(monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    categories = ["SNP", "hmer Indel 4", "Indel"]
    perf = pd.DataFrame({"tree_score": [0.1, 0.5], "recall": [0.9, 0.8], "precision": [0.8, 0.95], "f1": [0.85, 0.87]})
    perf_curve = dict.fromkeys(categories, perf)
    opt_res = dict.fromkeys(categories, {"recall": 0.85, "precision": 0.9})
    ReportUtils(verbosity=2, h5outfile="unused.h5").plot_performance(perf_curve, opt_res, categories, "test")
    assert categories == ["SNP", "hmer Indel 4", "Indel"]


def test_get_performance_error_types(tmp_path):
    n_variants = 20
    data = pd.DataFrame(
        {
            "indel": [False] * n_variants,
            "hmer_length": [0] * n_variants,
            "indel_length": [0] * n_variants,
            "call": ["TP"] * 15 + ["FP"] * 5,
            "base": ["TP"] * 15 + [np.nan] * 5,
            "tree_score": np.linspace(0, 1, n_variants),
            "filter": ["PASS"] * n_variants,
            "alleles": [("A", "C")] * n_variants,
            "gt_ultima": [(0, 1)] * n_variants,
            "gt_ground_truth": [(0, 1)] * 15 + [(0, 0)] * 5,
            "tp": [True] * 15 + [False] * 5,
            "fp": [False] * 15 + [True] * 5,
            "fn": [False] * n_variants,
            "error_type": [None] * n_variants,
        }
    )
    report_utils = ReportUtils(verbosity=1, h5outfile=str(tmp_path / "report.h5"))
    opt_tab, opt_res, perf_curve, _ = report_utils.get_performance(data, ["SNP"])
    assert opt_tab.loc["SNP", "recall"] == 1
    assert opt_tab.loc["SNP", "precision"] == 0.75
    assert opt_res["SNP"]["tp"] == 15
    assert len(perf_curve["SNP"]) == n_variants
