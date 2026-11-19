"""Coverage-oriented unit tests for ugbio_srsnv.srsnv_plotting_utils.

All tests run on small synthetic featuremaps (no real models; sklearn DummyClassifier stand-ins) so they
are fast. Large report figures are not written to disk: where a method renders a big figure we capture
the figure through ``_save_plt`` and assert on its axes instead.
"""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from sklearn.dummy import DummyClassifier  # noqa: E402
from ugbio_srsnv import srsnv_plotting_utils as spu  # noqa: E402
from ugbio_srsnv.srsnv_plotting_utils import SRSNVReport  # noqa: E402
from ugbio_srsnv.srsnv_report import _ModelWithTrainingResults  # noqa: E402
from ugbio_srsnv.srsnv_utils import PE_DUPLEX_GROUPS, prob_to_phred  # noqa: E402

RESOURCES = Path(__file__).parent.parent / "resources"
BASE_METADATA = RESOURCES / "402572-CL10377.srsnv_metadata.json"
H5_NAME = "t.single_read_snv.applicationQC.h5"
BASES = np.array(list("ACGT"))


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ───────────────────────────── synthetic builders ─────────────────────────────


def _base_df(n=1200, seed=0, *, n_folds=2, holdout_frac=0.15):
    """Synthetic SNV featuremap with separable TP/FP probabilities, 2 CV folds and a NaN-fold holdout."""
    rng = np.random.default_rng(seed)
    label = rng.random(n) < 0.5
    p = np.clip(np.where(label, rng.beta(8, 2, n), rng.beta(2, 8, n)), 1e-6, 1 - 1e-6)
    fm = pd.DataFrame({"label": label, "prob_orig": p, "MQUAL": prob_to_phred(p, max_value=60)})
    fm["SNVQ"] = np.clip(fm["MQUAL"] * 1.5 + 20, 0, 90)
    fold = rng.integers(0, n_folds, n).astype(float)
    fold[rng.random(n) < holdout_frac] = np.nan
    fm["fold_id"] = fold
    for k in range(n_folds):
        fm[f"prob_fold_{k}"] = np.clip(p + rng.normal(0, 0.05, n), 1e-6, 1 - 1e-6)
    ref = rng.integers(0, 4, n)
    alt = (ref + rng.integers(1, 4, n)) % 4
    fm["REF"] = BASES[ref]
    fm["ALT"] = BASES[alt]
    fm["X_PREV1"] = BASES[rng.integers(0, 4, n)]
    fm["X_NEXT1"] = BASES[rng.integers(0, 4, n)]
    fm["REV"] = rng.integers(0, 2, n)
    fm["RL"] = rng.integers(100, 200, n)
    fm["DP"] = rng.integers(20, 40, n)
    fm["INDEX"] = rng.integers(0, 100, n)
    fm["BCSQ"] = rng.uniform(0, 100, n)
    return fm


def _add_pe_duplex(fm, seed=1):
    rng = np.random.default_rng(seed)
    n = len(fm)
    fm = fm.copy()
    fm["dx"] = rng.choice([np.nan, 1.0], n)
    fm["cs_n_crossing"] = rng.integers(1, 5, n)
    fm["cs_n_supporting"] = np.minimum(fm["cs_n_crossing"], rng.integers(0, 5, n))
    fm["cs_family_size"] = rng.integers(1, 6, n)
    fm["cs_n_pe_pairs"] = rng.integers(0, 3, n)
    return fm


def _add_consensus(fm, seed=2):
    rng = np.random.default_rng(seed)
    fm = fm.copy()
    fm["fs"] = rng.integers(0, 4, len(fm))
    fm["rs"] = rng.integers(0, 4, len(fm))
    return fm


def _add_mixed(fm, seed=3):
    rng = np.random.default_rng(seed)
    n = len(fm)
    fm = fm.copy()
    fm["is_mixed_start"] = rng.random(n) < 0.5
    fm["is_mixed_end"] = rng.random(n) < 0.5
    fm["is_mixed"] = fm["is_mixed_start"] & fm["is_mixed_end"]
    fm["read_group"] = pd.Categorical(
        np.where(fm["is_mixed_start"], "Mixed", "Non-mixed"), categories=["Non-mixed", "Mixed"], ordered=True
    )
    fm["st"] = rng.choice(["MIXED", "PLUS", "MINUS", "END_UNREACHED"], n)
    fm["et"] = rng.choice(["MIXED", "PLUS", "MINUS", "END_UNREACHED"], n)
    return fm


def _add_indels(fm, *, hmer=True, non_hmer=True, seed=4, n_hmer=300, n_non_hmer=300):
    """Mark rows as hmer / non-hmer indels with the snvfind context fields the indel sections read."""
    rng = np.random.default_rng(seed)
    n = len(fm)
    fm = fm.copy()
    vt = np.array(["snv"] * n, dtype=object)
    xic = np.array([None] * n, dtype=object)
    start = 0
    if hmer:
        vt[start : start + n_hmer] = "hmer_indel"
        xic[start : start + n_hmer] = rng.choice(["ins", "del"], n_hmer)
        start += n_hmer
    if non_hmer:
        vt[start : start + n_non_hmer] = "non_hmer_indel"
        xic[start : start + n_non_hmer] = rng.choice(["ins", "del"], n_non_hmer)
    fm["variant_type"] = vt
    fm["X_IC"] = xic
    base_idx = rng.integers(0, 4, n)
    fm["X_HMER_BASE"] = BASES[base_idx]
    fm["X_HMER_PRE"] = BASES[(base_idx + rng.integers(1, 4, n)) % 4]
    fm["X_HMER_POST"] = BASES[(base_idx + rng.integers(1, 4, n)) % 4]
    fm["X_HMER_RUN"] = rng.integers(1, 16, n)
    fm["X_IL"] = np.where(vt == "non_hmer_indel", rng.integers(-8, 9, n), np.nan)
    motifs = ["AC", "AG", "CAG", "TTA", "ACGT", "AAAAT"]
    ru = np.where(vt == "non_hmer_indel", rng.choice([*motifs, "."], n), ".")
    fm["RU"] = ru
    fm["RPA"] = np.where(ru != ".", rng.integers(1, 9, n), np.nan)
    fm["STR"] = np.where(ru != ".", 1, np.nan)
    return fm


def _write_metadata(tmp_path, **overrides):
    meta = json.loads(BASE_METADATA.read_text())
    meta["features"] = [
        {"name": "RL", "type": "int"},
        {"name": "BCSQ", "type": "float"},
        {"name": "REF", "type": "c", "values": {"A": 0, "C": 1, "G": 2, "T": 3}},
    ]
    meta.update(overrides)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "meta.json"
    path.write_text(json.dumps(meta))
    return str(path), meta


def _dummy_models(n_folds=2):
    return [DummyClassifier(strategy="prior").fit([[0], [1]], [0, 1]) for _ in range(n_folds)]


def _report(fm, tmp_path, *, mode=None, n_folds=2, models=None, base_name="t.", extra_params=None, **kwargs):
    meta_path, _ = _write_metadata(tmp_path)
    params = {
        "num_CV_folds": n_folds,
        "numerical_features": ["RL", "BCSQ"],
        "categorical_features_names": ["REF"],
    }
    if mode:
        params["report_mode"] = mode
    params.update(extra_params or {})
    return SRSNVReport(
        models=models if models is not None else _dummy_models(n_folds),
        data_df=fm.copy(),
        params=params,
        out_path=str(tmp_path),
        srsnv_metadata=meta_path,
        base_name=base_name,
        raise_exceptions=True,
        rng=np.random.default_rng(0),
        **kwargs,
    )


def _h5_keys(report):
    if not os.path.exists(report.output_h5_filename):
        return set()
    with pd.HDFStore(report.output_h5_filename, "r") as store:
        return set(store.keys())


def _capture_figs(report):
    """Replace report._save_plt with a capturer (no PNG written); return the list of captured figures."""
    figs = []

    def _cap(output_filename=None, fig=None, **_kwargs):
        figs.append(fig if fig is not None else plt.gcf())

    report._save_plt = _cap
    return figs


class FakeXGBClassifier:
    """Minimal XGB-like classifier (passes SRSNVReport's model validation) with configurable evals_result."""

    def __init__(self, evals=None):
        self._evals = evals if evals is not None else {}

    def predict_proba(self, x):
        return np.tile([0.5, 0.5], (len(x), 1))

    def evals_result(self):
        return self._evals


# ───────────────────────────── module-level helpers ─────────────────────────────


def test_signif_rounds_to_significant_digits_and_keeps_zero_nan():
    out = spu.signif(np.array([123456.0, 0.00123456, 0.0, np.nan, -98.765]), 3)
    np.testing.assert_allclose(out[[0, 1, 2, 4]], [123000.0, 0.00123, 0.0, -98.8])
    assert np.isnan(out[3])


def test_list_of_jagged_lists_to_array_pads_with_fill():
    arr = spu.list_of_jagged_lists_to_array([[1, 2, 3], [4]], fill=-1)
    np.testing.assert_array_equal(arr, [[1, 2, 3], [4, -1, -1]])


def test_list_auc_to_qual_is_phred_of_auc():
    quals = spu.list_auc_to_qual([0.9, 0.99], max_value=100)
    np.testing.assert_allclose(quals, [10.0, 20.0])


@pytest.mark.parametrize(
    ("x", "expected_x", "expected_xlim"),
    [
        ([], [], None),
        ([3], [2.5, 3, 3.5], (2.5, 3.5)),
        ([0, 2, 4], [-1, 0, 2, 4, 5], (-1, 5)),
    ],
)
def test_plot_extended_step_and_fill_between_extend_edges(x, expected_x, expected_xlim):
    fig, ax = plt.subplots()
    y = list(range(len(x)))
    line = spu.plot_extended_step(x, y, ax)
    np.testing.assert_allclose(line.get_xdata(), expected_x)
    spu.plot_extended_fill_between(x, y, [v + 1 for v in y], ax)
    if expected_xlim is not None:
        np.testing.assert_allclose(ax.get_xlim(), expected_xlim)
    assert len(ax.collections) == 1


def test_plot_box_and_line_sets_default_alphas():
    fm = pd.DataFrame({"x": [1, 2, 3], "m": [5, 6, 7], "lo": [4, 5, 6], "hi": [6, 7, 8]})
    _, ax = plt.subplots()
    poly, line = spu.plot_box_and_line(fm, "x", "m", "lo", "hi", ax=ax)
    assert poly.get_alpha() == pytest.approx(0.2)
    assert line.get_alpha() == pytest.approx(0.7)


def test_discretized_bin_edges_are_half_integer_with_integer_width():
    a = np.array([0, 1, 2, 3, 5, 8, 10])
    edges = spu.discretized_bin_edges(a, bins=5)
    assert edges[0] == pytest.approx(-0.5)
    widths = np.diff(edges)
    assert np.allclose(widths, widths[0]) and widths[0] == pytest.approx(round(widths[0]))
    assert edges[-1] >= a.max()
    # explicit discretization size and range
    edges2 = spu.discretized_bin_edges(a, discretization_size=2, bins=5, full_range=(0, 10))
    assert np.allclose(np.diff(edges2) % 2, 0)


def test_filter_valid_queries_drops_failing_queries():
    fm = pd.DataFrame({"a": [1, 2, 3]})
    valid = spu.filter_valid_queries(fm, {"ok": "a > 1", "bad": "missing_col > 1"}, verbose=True)
    assert valid == {"ok": "a > 1"}


def test_precision_and_recall_score_with_mask():
    y_pred = np.array([1, 1, 0, 0])
    y_true = np.array([1, 0, 1, 0])
    assert spu.precision_score_with_mask(y_pred, y_true, np.array([True, True, False, False])) == 0.5
    # no positive prediction under mask -> 1 by convention
    assert spu.precision_score_with_mask(y_pred, y_true, np.array([False, False, True, True])) == 1
    assert spu.recall_score_with_mask(y_pred, y_true, np.array([True, False, True, False])) == 0.5
    # no true positive under mask -> 1 by convention
    assert spu.recall_score_with_mask(y_pred, y_true, np.array([False, True, False, True])) == 1


def test_precision_recall_curve_cumulative_and_log():
    score = pd.Series([0.2, 1.5, 1.7, 2.1, 2.9, 3.5])
    y_true = np.array([False, False, True, True, True, True])
    quals, recalls = spu.precision_recall_curve(score, max_score=5, y_true=y_true, cumulative=True)
    assert len(quals) == len(recalls) == 5
    # cumulative threshold 0 keeps everything: precision 4/6 -> qual -10log10(1/3)
    assert quals[0] == pytest.approx(-10 * np.log10(1 - 4 / 6))
    # threshold 2 keeps only TPs -> precision 1 -> qual capped at max_score
    assert quals[2] == 5
    precisions, _ = spu.precision_recall_curve(score, max_score=4, y_true=y_true, apply_log_trans=False)
    # non-cumulative bin floor==1 has (1.5 FP, 1.7 TP) -> precision 0.5
    assert precisions[1] == pytest.approx(0.5)


def test_create_data_for_report_splits_tp_fp():
    rng = np.random.default_rng(0)
    fm = pd.DataFrame({"label": rng.random(50) < 0.4, "ML_qual_0": rng.uniform(0, 5, 50)})
    fm["ML_qual_1"] = rng.uniform(0, 5, 50)
    clf = SimpleNamespace(feature_names_in_=np.array(["f1", "f2"]))
    out_df, df_tp, df_fp, max_score, features, fprs, recalls = spu.create_data_for_report([clf], fm)
    assert out_df is fm
    assert len(df_tp) == fm["label"].sum() and len(df_fp) == (~fm["label"]).sum()
    assert features == ["f1", "f2"]
    assert max_score == 5
    assert set(fprs) == {0, 1} and len(fprs[1][0]) == max_score and len(recalls[0][0]) == max_score


def test_get_data_subsets_and_fpr_recalls():
    fm = pd.DataFrame(
        {
            "is_cycle_skip": [1, 0, 1, 0, 1, 0],
            "is_mixed": [1, 1, 0, 0, 1, 0],
            "label": [True, False, True, False, True, True],
            "ML_qual_1": [3.0, 1.0, 2.5, 0.5, 2.0, 1.5],
        }
    )
    mixed = spu.get_data_subsets(fm.copy(), is_mixed_flag=True)
    assert {k: len(v) for k, v in mixed.items()} == {
        "mixed cycle skip": 2,
        "mixed non cycle skip": 1,
        "non mixed non cycle skip": 2,
        "non mixed cycle skip": 1,
    }
    plain = spu.get_data_subsets(fm.copy(), is_mixed_flag=False)
    assert {k: len(v) for k, v in plain.items()} == {"cycle skip": 3, "non cycle skip": 3}
    fpr, rec = spu.get_fpr_recalls_subsets(plain, max_score=3)
    assert set(fpr) == set(rec) == {"cycle skip", "non cycle skip"}
    assert all(len(v) == 3 for v in fpr.values())


def test_get_plot_paths_order():
    paths = spu._get_plot_paths("/o", "b.")
    assert len(paths) == 10
    assert paths[0] == "/o/b.FQ_vs_recall" and paths[-1] == "/o/b.calibration_fn_with_hist"
    assert paths[6] == "/o/b.qual_per_"


def _legacy_df(n=200, seed=0):
    rng = np.random.default_rng(seed)
    fm = pd.DataFrame({"label": rng.random(n) < 0.5})
    fm["ML_qual_1"] = np.where(fm["label"], rng.uniform(2, 10, n), rng.uniform(0, 6, n))
    fm["ML_qual_0"] = 10 - fm["ML_qual_1"]
    fm["ML_prediction_1"] = (fm["ML_qual_1"] > 5).astype(int)
    fm["flag"] = rng.random(n) < 0.3
    fm["cat"] = pd.Categorical(rng.choice(list("ABCDE"), n))
    fm["num"] = rng.normal(0, 1, n)
    fm["is_cycle_skip"] = rng.integers(0, 2, n)
    fm["is_mixed"] = rng.integers(0, 2, n)
    return fm


def _record_savefig(monkeypatch):
    """Record plt.savefig target paths instead of rendering 300-dpi PNGs (keeps the tests fast)."""
    saved = []
    monkeypatch.setattr(plt, "savefig", lambda fname, *_a, **_kw: saved.append(os.path.basename(str(fname))))
    return saved


def test_legacy_plot_functions_write_pngs(tmp_path, monkeypatch):
    """Exercise the legacy (pre-DNN) plotting helpers; each must save its PNG and plot the expected series."""
    saved = _record_savefig(monkeypatch)
    fm = _legacy_df()
    labels = {0: "FP", 1: "TP"}
    _, _, _, max_score, _, fprs, recalls = spu.create_data_for_report(
        [SimpleNamespace(feature_names_in_=np.array(["num"]))], fm
    )
    out = str(tmp_path / "p")

    spu.plot_confusion_matrix(fm, output_filename=out + "_cm")
    assert [t.get_text() for t in plt.gca().get_xticklabels()] == ["FP", "TP"]

    spu.plot_observed_vs_measured_qual(labels, fprs, max_score, output_filename=out + "_obs")
    assert len(plt.gca().get_lines()) == 3  # 2 labels + identity line

    spu.plot_qual_density(labels, recalls, max_score, output_filename=out + "_dens")
    assert len(plt.gca().get_lines()) == 2

    spu.plot_precision_recall_vs_qual_thresh(fm, labels, max_score, output_filename=out + "_pr")
    assert len(plt.gca().get_lines()) == 2

    spu.plot_ml_qual_hist(labels, fm, max_score, output_filename=out + "_hist")
    assert plt.gca().get_yscale() == "log"

    spu.plot_qual_per_feature(labels, ["flag", "cat", "num"], fm, output_filename=out + "_feat_")

    subsets = spu.get_data_subsets(fm.copy(), is_mixed_flag=True)
    fpr_d, rec_d = spu.get_fpr_recalls_subsets(subsets, max_score=max_score)
    spu.plot_subsets_hists(labels, subsets, max_score, output_filename=out + "_sub_")
    spu.plot_mixed_fpr(fpr_d, max_score, output_filename=out + "_mfpr")
    assert len(plt.gca().get_lines()) == len(fpr_d) + 1
    spu.plot_mixed_recall(rec_d, max_score, output_filename=out + "_mrec")
    assert len(plt.gca().get_lines()) == len(rec_d)

    expected = [
        "p_cm.png",
        "p_obs.png",
        "p_dens.png",
        "p_pr.png",
        "p_hist.png",
        "p_feat_flag.png",
        "p_feat_cat.png",
        "p_feat_num.png",
        "p_sub_mixed_cycle_skip.png",
        "p_mfpr.png",
        "p_mrec.png",
    ]
    for name in expected:
        assert name in saved, name
    assert len([n for n in saved if n.startswith("p_sub_")]) == 4


def test_plot_precision_recall_log_scale():
    plt.figure()
    spu.plot_precision_recall([[1, 2, 3, 4], [2, 3, 4, 5]], ["a", "b"], max_score=3, log_scale=True)
    ax = plt.gca()
    assert ax.get_yscale() == "log"
    assert [len(line.get_xdata()) for line in ax.get_lines()] == [3, 3]


# ───────────────────────────── LoD simulation / stats ─────────────────────────────


def _lod_inputs(tmp_path, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    fm = pd.DataFrame({"label": (rng.random(n) < 0.5).astype(int), "BCSQ": rng.uniform(0, 120, n)})
    fm["EDIST"] = rng.integers(0, 8, n)
    fm["INDEX"] = rng.integers(0, 150, n)
    fm["RL"] = 150
    fm["MQUAL"] = np.where(fm["label"] == 1, rng.uniform(5, 40, n), rng.uniform(0, 15, n))
    bed = tmp_path / "regions.bed"
    bed.write_text("chr1\t0\t1000000\nchr2\t100\t200100\n")
    cvg = [0] * 10 + [100] * 40 + [5] * 10
    sorter = tmp_path / "sorter.json"
    sorter.write_text(json.dumps({"base_coverage": {"Genome": cvg}, "cvg": cvg, "mapq": [5] * 60 + [95]}))
    filters = {
        "no_filter": "BCSQ >= 0",
        "HQ_SNV": "BCSQ >= 79 and EDIST <= 5",
        "ML_qual_5": "MQUAL >= 5",
        "ML_qual_10": "MQUAL >= 10",
        "ML_qual_20": "MQUAL >= 20",
        "mixed_ML_qual_10": "MQUAL >= 10 and EDIST <= 3",
        "never": "BCSQ > 1000",  # TP retention 0 -> dropped
        "broken": "no_such_column > 0",  # raises -> filtered out by filter_valid_queries
    }
    return fm, str(bed), str(sorter), filters


def test_retention_noise_and_mrd_lod_simulation(tmp_path):
    fm, bed, sorter, filters = _lod_inputs(tmp_path)
    df_mrd, valid_filters, _lod_label, c_lod, norm = spu.retention_noise_and_mrd_lod_simulation(
        fm,
        single_sub_regions=bed,
        sorter_json_stats_file=sorter,
        fp_featuremap_entry_number=10_000,
        lod_filters=filters,
    )
    assert "broken" not in valid_filters
    assert c_lod == "LoD_90"
    assert "never" not in df_mrd.index  # low TP retention removed
    assert norm["n_bases_in_region"] == 1_200_000
    assert df_mrd.loc["no_filter", spu.FP_READ_RETENTION_RATIO] == pytest.approx(1.0)
    assert norm["read_filter_correction_factor"] == 1  # no X_FILTERED_COUNT / X_READ_COUNT columns


def test_lod_simulation_read_filter_correction_from_count_columns(tmp_path):
    # Regression: FeatureMapFields had no READ_COUNT / FILTERED_COUNT members, so this raised AttributeError on
    # every call. With the X_FILTERED_COUNT / X_READ_COUNT columns the correction is sum(filtered+1)/sum(read).
    fm, bed, sorter, filters = _lod_inputs(tmp_path)
    fm["X_FILTERED_COUNT"] = 1
    fm["X_READ_COUNT"] = 4
    _, _, _, _, norm = spu.retention_noise_and_mrd_lod_simulation(
        fm,
        single_sub_regions=bed,
        sorter_json_stats_file=sorter,
        fp_featuremap_entry_number=10_000,
        lod_filters=filters,
    )
    assert norm["read_filter_correction_factor"] == pytest.approx(0.5)  # (1 + 1) / 4 over the FP rows
    assert norm["ratio_filtered_prior_to_featuremap"] == pytest.approx(norm["ratio_of_reads_over_mapq"] * 0.5)


def _df_mrd():
    """Hand-built MRD-simulation table (the shape retention_noise_and_mrd_lod_simulation returns)."""
    return pd.DataFrame(
        {
            "LoD_90": [1e-4, 8e-5, 5e-5, 2e-5, 3e-5, 6e-5, 7e-5],
            spu.RESIDUAL_SNV_RATE: [1e-4, 5e-5, 1e-5, 1e-6, 2e-6, 3e-5, 4e-5],
            spu.TP_READ_RETENTION_RATIO: [0.95, 0.9, 0.8, 0.5, 0.6, 0.85, 0.7],
        },
        index=[
            "no_filter",
            "HQ_SNV",
            "ML_qual_5",
            "ML_qual_20",
            "mixed_ML_qual_10",
            "HQ_SNV_mixed_only",
            "CSKP_mixed_only",
        ],
    )


def test_calculate_lod_stats_and_lod_plots(tmp_path, monkeypatch):
    saved = _record_savefig(monkeypatch)
    df_mrd = _df_mrd()
    h5 = str(tmp_path / "lod.h5")
    best = spu.calculate_lod_stats(df_mrd, output_h5=h5, lod_column="LoD_90")
    assert best == "ML_qual_20"
    stats = pd.read_hdf(h5, "lod_stats")
    assert stats["best_ML_LoD_filter"] == "ML_qual_20"
    assert stats["best_mixed_ML_LoD_filter"] == "mixed_ML_qual_10"
    assert stats["LoD-no_filter"] == pytest.approx(1e-4)
    assert stats[f"{spu.TP_READ_RETENTION_RATIO}-HQ_SNV"] == pytest.approx(0.9)
    assert "LoD-ML_qual_5" not in stats.index  # ML rows are summarized, not listed
    bases_over = pd.read_hdf(h5, "bases_over_qual_threshold")
    assert bases_over.index.name == "QualThreshold"
    # ML nodes: phred 50 -> 0.8, phred 60 -> 0.5; below the min qual -> 1, above the max -> 0
    assert bases_over.loc[50] == pytest.approx(0.8)
    assert bases_over.loc[55] == pytest.approx(0.65)
    assert bases_over.loc[40] == 1 and bases_over.loc[70] == 0

    filters = {k: "x" for k in df_mrd.index}
    lod_label = "LoD label"
    spu.plot_LoD(
        df_mrd,
        lod_label,
        "LoD_90",
        filters,
        "v1",
        best,
        output_filename=str(tmp_path / "lod"),
        extra_filters={"x": "HQ_SNV"},
    )
    assert "lod.png" in saved
    legend_texts = [t.get_text() for t in plt.gca().get_legend().get_texts()]
    assert any(t.startswith("ML model, best LoD: 2.0E-5") for t in legend_texts)
    assert any("mixed only" in t for t in legend_texts)  # ppmSeq adapter adds the mixed-only series
    spu.plot_LoD(df_mrd, lod_label, "LoD_90", filters, "other_adapter", best)
    legend_texts = [t.get_text() for t in plt.gca().get_legend().get_texts()]
    assert not any("mixed only" in t for t in legend_texts)

    spu.plot_LoD_vs_qual(df_mrd, "LoD_90", output_filename=str(tmp_path / "lod_vs_qual"))
    assert saved[-1] == "lod_vs_qual.png"


def test_calculate_lod_stats_without_mixed_filters(tmp_path):
    df_mrd = _df_mrd().drop(index=["mixed_ML_qual_10"])
    h5 = str(tmp_path / "x.h5")
    assert spu.calculate_lod_stats(df_mrd, output_h5=h5, lod_column="LoD_90") == "ML_qual_20"
    stats = pd.read_hdf(h5, "lod_stats")
    assert pd.isna(stats["best_mixed_ML_LoD_filter"])


def test_create_srsnv_report_html_passes_indel_flags_and_paths(tmp_path):
    with patch.object(spu, "generate_report") as gen:
        spu.create_srsnv_report_html(
            out_path=str(tmp_path),
            out_basename="sample",
            srsnv_metadata_file="meta.json",
            split_mode="pe_duplex",
            display_suffix="pe_duplex",
            has_hmer_indels=True,
            has_non_hmer_indels=False,
            has_duplex_pe=True,
        )
    kwargs = gen.call_args.kwargs
    assert kwargs["output_report_html_path"] == tmp_path / "sample.report.html"
    params = kwargs["parameters"]
    assert params["has_hmer_indels"] is True and params["has_non_hmer_indels"] is False
    assert params["has_duplex_pe"] is True
    assert params["split_mode"] == "pe_duplex"
    assert params["srsnv_qc_h5_file"] == str(tmp_path / "sample.single_read_snv.applicationQC.h5")
    assert params["non_hmer_indel_str_context_plot"] == str(tmp_path / "sample.non_hmer_indel_str_context")
    assert params["logit_histogram_hmer_indel"] == str(tmp_path / "sample.logit_histogram_hmer_indel")


# ───────────────────────────── SRSNVReport: construction ─────────────────────────────


def test_report_init_validation_errors(tmp_path):
    fm = _base_df(100)
    with pytest.raises(TypeError, match="list of models"):
        _report(fm, tmp_path, models=tuple(_dummy_models(2)))
    with pytest.raises(ValueError, match="not a classifier"):
        _report(fm, tmp_path, models=[object(), object()])
    meta_path, _ = _write_metadata(tmp_path)
    with pytest.raises(TypeError, match="not a DataFrame"):
        SRSNVReport(
            models=_dummy_models(2), data_df=fm.to_dict(), params={}, out_path=str(tmp_path), srsnv_metadata=meta_path
        )
    with pytest.raises(ValueError, match="statistics_h5_file is required"):
        _report(fm, tmp_path, statistics_json_file="x.json")


def test_report_init_adds_columns_and_rng_default(tmp_path):
    meta_path, _ = _write_metadata(tmp_path)
    fm = _base_df(200)
    rep = SRSNVReport(
        models=_dummy_models(2),
        data_df=fm.copy(),
        params={"num_CV_folds": 2},
        out_path=str(tmp_path),
        srsnv_metadata=meta_path,
        base_name="t.",
    )
    assert isinstance(rep.rng, np.random.Generator)
    assert rep.report_mode.value == "none"
    np.testing.assert_array_equal(rep.data_df["is_forward"].to_numpy(), fm["REV"].to_numpy() != 1)
    assert rep.data_df["trinuc_context_with_alt"].iloc[0] == (
        fm["X_PREV1"].iloc[0] + fm["REF"].iloc[0] + fm["X_NEXT1"].iloc[0] + fm["ALT"].iloc[0]
    )
    # training predictions: each row's own-fold probability is excluded
    assert {"ML_prob_train", "ML_logit_train", "ML_logit_test"} <= set(rep.data_df.columns)


def test_report_without_base_name_has_clean_h5_name(tmp_path):
    # Regression: base_name=None (the default) produced "Nonesingle_read_snv.applicationQC.h5" and a
    # TypeError in the run-info tables (base_name[:-1]).
    rep = _report(_base_df(100), tmp_path, base_name=None)
    assert rep.params["data_name"] == ""
    assert os.path.basename(rep.output_h5_filename) == "single_read_snv.applicationQC.h5"


def test_display_groups_and_palette_per_scheme(tmp_path):
    pe = _report(_add_pe_duplex(_base_df(300)), tmp_path)
    assert pe._display_groups() == PE_DUPLEX_GROUPS
    assert pe._variant_palette()["duplex PE"] == "tab:green"
    assert pe.tag_axis == "duplex overlap"
    cons = _report(_add_consensus(_base_df(300)), tmp_path / "c")
    pal = cons._variant_palette()
    assert list(pal) == ["single read", "consensus, one strand", "consensus, duplex"]
    assert len(set(pal.values())) == 3  # seaborn fallback gives distinct colors


# ───────────────────────────── SRSNVReport: statistics ─────────────────────────────


def test_get_snvq_and_recall_monotone(tmp_path):
    rep = _report(_base_df(500), tmp_path)
    labels = rep.data_df["label"].to_numpy()
    mq = rep.data_df["MQUAL"].to_numpy()
    snvqs, recalls = rep._get_snvq_and_recall(labels, mq, base_snvq=1e-3, base_recall=0.8, ml_qual_max=10)
    assert len(snvqs) == len(recalls) == 11
    assert recalls[0] == pytest.approx(0.8 * ((mq > 0) & labels).sum() / labels.sum())
    assert np.all(np.diff(recalls) <= 0)
    cond = np.zeros_like(labels, dtype=bool)
    cond[:100] = True
    _, rec_c = rep._get_snvq_and_recall(labels, mq, base_snvq=1e-3, base_recall=0.8, condition=cond, ml_qual_max=3)
    assert rec_c[0] < recalls[0]


def test_tpr_fpr_and_filter_quality(tmp_path):
    rep = _report(_base_df(500), tmp_path)
    d = rep.data_df
    tprs, fprs = rep._calculate_tpr_fpr_for_thresholds([0, 10, 1000], recall0=0.5)
    assert tprs[0] == pytest.approx(0.5 * (d["label"] & (d["MQUAL"] > 0)).sum() / d["label"].sum())
    assert fprs[1] == pytest.approx(((~d["label"]) & (d["MQUAL"] > 10)).sum() / (~d["label"]).sum())
    assert tprs[2] == 0 and fprs[2] == 0
    fqs = rep._calculate_filter_quality([0.5, 0.0, 0.5], [0.05, 0.1, 10.0], base_error_rate=1e-3)
    assert fqs[0] == pytest.approx(prob_to_phred(1 - 0.1 * 1e-3, max_value=rep.max_qual))
    assert fqs[1] == 0  # tpr == 0 -> precision 0
    assert fqs[2] == pytest.approx(prob_to_phred(1 - 20 * 1e-3, max_value=rep.max_qual))


def test_get_dataset_sizes_multi_fold_with_other_ids(tmp_path):
    fm = _base_df(400)
    fm.loc[fm.index[:10], "fold_id"] = 5  # a fold id outside range(num_CV_folds)
    for k in range(2):
        fm.loc[fm.index[:10], f"prob_fold_{k}"] = 0.5
    rep = _report(fm, tmp_path)
    sizes = rep.get_dataset_sizes()
    assert sizes[("All", "Total")] == 400
    assert sizes[("All", "other")] == 10
    total = sizes[("All", "fold 0")] + sizes[("All", "fold 1")] + sizes[("All", "test only")] + sizes[("All", "other")]
    assert total == 400
    assert sizes[("TPs", "Total")] + sizes[("FPs", "Total")] == 400


def test_get_dataset_sizes_single_fold(tmp_path):
    fm = _base_df(300, n_folds=1, holdout_frac=0.3)
    fm["fold_id"] = np.where(fm["fold_id"].isna(), np.nan, 0.0)
    rep = _report(fm, tmp_path, n_folds=1)
    sizes = rep.get_dataset_sizes()
    assert set(sizes) == {(s, g) for s in ("All", "TPs", "FPs", "% TPs") for g in ("val", "test")}
    assert sizes[("All", "val")] + sizes[("All", "test")] == 300
    assert sizes[("% TPs", "val")] == pytest.approx(
        spu.signif(100 * sizes[("TPs", "val")] / sizes[("All", "val")], spu.SIG_DIGITS)
    )


def test_roc_auc_table_holdout_rows(tmp_path):
    """Holdout (NaN fold) above the size threshold adds holdout mean/std rows per group; a group whose
    holdout has a single label yields NaN on the first fold and is not recomputed for later folds."""
    fm = _add_pe_duplex(_base_df(1500, holdout_frac=0.3))
    rep = _report(fm, tmp_path)
    # make every holdout "duplex PE" read a TP -> holdout ROC AUC undefined for that group
    ho_pe = rep.data_df["fold_id"].isna() & (rep.data_df["pe_duplex_group"] == "duplex PE")
    rep.data_df.loc[ho_pe, "label"] = True
    rep.calc_roc_auc_table(holdout_fold_size_thresh=100)
    table = pd.read_hdf(rep.output_h5_filename, "roc_auc_table_pe_duplex")
    assert list(table.columns) == ["Total", *[f"{g} only" for g in PE_DUPLEX_GROUPS]]
    assert "ROC AUC on holdout mean" in table.index and "ROC AUC on holdout std" in table.index
    assert np.isfinite(table.loc["ROC AUC on holdout mean", "Total"])
    assert np.isnan(table.loc["ROC AUC on holdout mean", "duplex PE only"])
    assert np.isfinite(table.loc["ROC AUC on holdout mean", "simplex SE only"])
    # bare base key is written too (no explicit legacy variant in pe-duplex)
    assert "/roc_auc_table" in _h5_keys(rep)

    rep2 = _report(fm, tmp_path / "b")
    rep2.calc_roc_auc_table(holdout_fold_size_thresh=10**6)
    table2 = pd.read_hdf(rep2.output_h5_filename, "roc_auc_table")
    assert "ROC AUC on holdout mean" not in table2.index


def test_roc_auc_table_total_holdout_nan_stops_recompute(tmp_path):
    fm = _base_df(800, holdout_frac=0.3)
    rep = _report(fm, tmp_path)
    rep.data_df.loc[rep.data_df["fold_id"].isna(), "label"] = True  # whole holdout single-label
    rep.calc_roc_auc_table(holdout_fold_size_thresh=50)
    table = pd.read_hdf(rep.output_h5_filename, "roc_auc_table")
    assert np.isnan(table.loc["ROC AUC on holdout mean", "Total"])


def test_run_info_table_by_group_values(tmp_path):
    fm = _add_pe_duplex(_base_df(1200))
    fm["cs_n_cand"] = np.random.default_rng(0).integers(1, 3, len(fm))
    rep = _report(fm, tmp_path, extra_params={"pipeline_version": "1.2.3", "docker_image": "img:tag"})
    rep.plot_fq_recall(only_calculate=True)
    rep.calc_run_info_table()
    d = rep.data_df
    summary = pd.read_hdf(rep.output_h5_filename, "run_quality_summary_table_pe_duplex")
    tp = d[d["label"]]
    for grp in PE_DUPLEX_GROUPS:
        g_tp = tp[tp["pe_duplex_group"] == grp]
        assert summary[("Median SNVQ", grp)] == pytest.approx(spu.signif(g_tp["SNVQ"].median(), spu.SIG_DIGITS))
        # conditional recall = fraction of the group's TP reads with SNVQ >= 60
        assert summary[("Recall at SNVQ=60", grp)] == pytest.approx(
            spu.signif((g_tp["SNVQ"] >= 60).mean(), spu.SIG_DIGITS), rel=1e-3
        )
        g = d[d["pe_duplex_group"] == grp]
        assert summary[("Median CS candidates", grp)] == pytest.approx(g["cs_n_cand"].median())
    assert summary[("Median CS family size", "All reads")] == pytest.approx(d["cs_family_size"].median())
    assert ("Median dropped-cand support", "All reads") not in summary.index  # column absent -> row skipped
    run_info = pd.read_hdf(rep.output_h5_filename, "run_info_table")
    assert run_info[("Sample name", "")] == "t"
    assert run_info[("Pipeline version", "")] == "1.2.3"
    pct = run_info[("duplex PE training reads", "% of TP")]
    expected = spu.signif(100 * (tp["pe_duplex_group"] == "duplex PE").mean(), 3)
    assert pct == f"{expected}%"
    training = pd.read_hdf(rep.output_h5_filename, "training_info_table")
    assert training[("Number of CV folds", "")] == 2


def test_run_info_and_quality_tables_mixed_mode(tmp_path):
    rep = _report(_add_mixed(_base_df(1000)), tmp_path, mode="mixed")
    rep.plot_fq_recall(only_calculate=True)
    rep.calc_run_info_table()
    rep.calc_run_quality_table()
    keys = _h5_keys(rep)
    for k in (
        "/run_info_table",
        "/run_info_table_mixed_start",
        "/run_quality_summary_table",
        "/run_quality_summary_table_mixed_start",
        "/run_quality_table",
        "/run_quality_table_mixed_start",
        "/run_quality_table_display",
        "/FQ_recall_LoD",
        "/FQ_recall_LoD_mixed_start",
    ):
        assert k in keys, k
    d = rep.data_df
    legacy = pd.read_hdf(rep.output_h5_filename, "run_quality_summary_table")
    tp_both = d[d["label"] & d["is_mixed"]]
    assert legacy[("Median SNVQ", "Mixed, both ends")] == pytest.approx(
        spu.signif(tp_both["SNVQ"].median(), spu.SIG_DIGITS)
    )
    display = pd.read_hdf(rep.output_h5_filename, "run_quality_summary_table_mixed_start")
    tp_start = d[d["label"] & d["is_mixed_start"]]
    assert display[("Median SNVQ", "Mixed")] == pytest.approx(spu.signif(tp_start["SNVQ"].median(), spu.SIG_DIGITS))
    info_legacy = pd.read_hdf(rep.output_h5_filename, "run_info_table")
    expected = spu.signif(100 * (d["is_mixed"] & d["label"]).sum() / d["label"].sum(), 3)
    assert info_legacy[("Mixed training reads", "% of TP")] == f"{expected}%"
    qdisp = pd.read_hdf(rep.output_h5_filename, "run_quality_table_display")
    assert set(qdisp.columns.get_level_values(2)) == {"", "overall", "mixed", "non-mixed"}
    # legacy FQ conditions: all reads + mixed both ends + mixed start
    fq = pd.read_hdf(rep.output_h5_filename, "FQ_recall_LoD")
    assert {"recall_all reads", "recall_mixed both ends", "recall_mixed start"} <= set(fq.columns)


def test_quality_per_ppmseq_tags_mixed_crosstab(tmp_path):
    rep = _report(_add_mixed(_base_df(800)), tmp_path, mode="mixed")
    figs = _capture_figs(rep)
    rep.quality_per_ppmseq_tags(output_filename="x")
    d = rep.data_df[rep.data_df["label"]]
    summary = pd.read_hdf(rep.output_h5_filename, "ppmseq_category_quality_table_mixed_start")
    assert "END_UNREACHED" not in summary.index
    assert summary.index.name == "ppmSeq tag"
    assert summary.loc["MIXED", "Median SNVQ"] == pytest.approx(
        spu.signif(d.loc[d["st"] == "MIXED", "SNVQ"].median(), 3)
    )
    qty = pd.read_hdf(rep.output_h5_filename, "ppmseq_category_quantity_table")
    # quantity table excludes END_UNREACHED start rows and is renormalized to 100%
    assert qty.sum() == pytest.approx(100.0)
    assert len(figs) == 1


def test_quality_per_ppmseq_tags_mixed_without_tag_columns(tmp_path):
    fm = _add_mixed(_base_df(400)).drop(columns=["st", "et"])
    rep = _report(fm, tmp_path, mode="mixed")
    _capture_figs(rep)
    rep.quality_per_ppmseq_tags()
    summary = pd.read_hdf(rep.output_h5_filename, "ppmseq_category_quality_table_mixed_start")
    assert len(summary) == 1  # no tag columns -> a single all-NaN tag row
    assert summary["% of reads"].iloc[0] == pytest.approx(100.0)


def test_quality_per_split_group_pe_duplex(tmp_path):
    rep = _report(_add_pe_duplex(_base_df(800)), tmp_path)
    figs = _capture_figs(rep)
    rep.quality_per_ppmseq_tags(output_filename="x")
    summary = pd.read_hdf(rep.output_h5_filename, "ppmseq_category_quality_table_pe_duplex")
    assert list(summary.index) == PE_DUPLEX_GROUPS
    assert summary["% of reads"].sum() == pytest.approx(100.0, abs=2)
    ax = figs[0].axes[0]
    assert len(ax.tables) == 1


def test_run_quality_table_by_group(tmp_path):
    rep = _report(_add_consensus(_base_df(800)), tmp_path)
    rep.calc_run_quality_table()
    table = pd.read_hdf(rep.output_h5_filename, "run_quality_table_strand_support")
    d = rep.data_df
    assert table.loc["SNVQ_TP", "50%"] == pytest.approx(spu.signif(d.loc[d["label"], "SNVQ"].median(), spu.SIG_DIGITS))
    assert "SNVQ_TP_consensus, duplex" in table.index
    assert "count" not in table.columns
    disp = pd.read_hdf(rep.output_h5_filename, "run_quality_table_display")
    assert set(disp.columns.get_level_values(2)) == {"", "overall", *rep._display_groups()}


def test_fq_recall_plot_writes_display_curves(tmp_path):
    rep = _report(_add_pe_duplex(_base_df(1000)), tmp_path)
    figs = _capture_figs(rep)
    rep.plot_fq_recall(output_filename="fq")
    fq = pd.read_hdf(rep.output_h5_filename, "FQ_recall_LoD_pe_duplex")
    assert "recall_all reads" in fq.columns
    present = [g for g in PE_DUPLEX_GROUPS if (rep.data_df["pe_duplex_group"] == g).any()]
    for g in present:
        assert f"FQ_{g}" in fq.columns
    assert len(figs[0].axes[0].get_lines()) == 1 + len(present)
    # recall never exceeds the prefilter base recall
    assert (fq["recall_all reads"] <= rep.base_recall + 1e-9).all()


# ───────────────────────────── SRSNVReport: training / SHAP / figures ─────────────────────────────


def _evals(n_steps=5):
    return {
        "validation_0": {"logloss": list(np.linspace(0.6, 0.3, n_steps)), "auc": list(np.linspace(0.8, 0.95, n_steps))},
        "validation_1": {
            "logloss": list(np.linspace(0.65, 0.4, n_steps)),
            "auc": list(np.linspace(0.78, 0.9, n_steps)),
        },
    }


def test_training_progress_plot_writes_h5(tmp_path):
    models = [FakeXGBClassifier(_evals(5)), FakeXGBClassifier(_evals(3))]
    rep = _report(_base_df(200), tmp_path, models=models)
    figs = _capture_figs(rep)
    rep.training_progress_plot(output_filename="tp", ylims=[(0, 1), (0, 20)])
    tp = pd.read_hdf(rep.output_h5_filename, "training_progress")
    assert tp.shape == (8, 5)  # (2 metrics x 2 sets x 2 folds) rows, max steps columns
    assert np.isnan(tp.loc[("logloss", "train", "fold1")].iloc[4])  # jagged fold padded with NaN
    assert tp.loc[("auc", "val", "fold0")].iloc[0] == pytest.approx(prob_to_phred(0.78))
    assert figs[0].axes[0].get_ylim() == (0, 1)


def test_training_progress_plot_mlogloss_single_fold(tmp_path):
    ev = _evals(4)
    for k in ("validation_0", "validation_1"):
        ev[k]["mlogloss"] = ev[k].pop("logloss")
    rep = _report(_base_df(200, n_folds=1), tmp_path, models=[FakeXGBClassifier(ev)], n_folds=1)
    _capture_figs(rep)
    rep.training_progress_plot()
    tp = pd.read_hdf(rep.output_h5_filename, "training_progress")
    assert tp.shape == (4, 4)


def test_training_progress_plot_skips_or_raises(tmp_path):
    rep = _report(_base_df(200), tmp_path, models=[FakeXGBClassifier({}), FakeXGBClassifier({})])
    rep.training_progress_plot()
    assert "/training_progress" not in _h5_keys(rep)
    bad = {"validation_0": {"error": [0.1]}, "validation_1": {"error": [0.1]}}
    rep2 = _report(_base_df(200), tmp_path / "b", models=[FakeXGBClassifier(bad), FakeXGBClassifier(bad)])
    with pytest.raises(ValueError, match="Neither 'logloss' nor 'mlogloss'"):
        rep2.training_progress_plot()


def test_exception_handler_swallows_when_not_raising(tmp_path):
    bad = {"validation_0": {"error": [0.1]}, "validation_1": {"error": [0.1]}}
    rep = _report(_base_df(200), tmp_path, models=[FakeXGBClassifier(bad), FakeXGBClassifier(bad)])
    spu.exception_config.raise_exception = False
    try:
        assert rep.training_progress_plot() is None
    finally:
        spu.exception_config.raise_exception = True


def test_shap_skipped_for_dummy_models(tmp_path):
    models = [_ModelWithTrainingResults(m) for m in _dummy_models(2)]
    rep = _report(_base_df(200), tmp_path, models=models)
    with patch.object(SRSNVReport, "_create_shap_plotter") as mk:
        rep.calc_and_plot_shap_values()
    mk.assert_not_called()
    assert "/mean_abs_SHAP_scores" not in _h5_keys(rep)


def test_shap_values_written_with_mocked_plotter(tmp_path):
    rep = _report(_base_df(200), tmp_path, models=[FakeXGBClassifier(), FakeXGBClassifier()])
    _capture_figs(rep)
    x_sample = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
    shap_values = np.array([[0.1, -0.5, 9.0], [-0.3, 0.5, 9.0], [0.2, -0.5, 9.0]])  # last col = base value
    plotter = SimpleNamespace(plot_both=lambda **_kw: (plt.figure(), plt.figure(), shap_values, x_sample))
    with patch.object(SRSNVReport, "_create_shap_plotter", return_value=plotter):
        rep.calc_and_plot_shap_values()
    scores = pd.read_hdf(rep.output_h5_filename, "mean_abs_SHAP_scores")
    assert list(scores.index) == ["f2", "f1"]
    assert scores["f2"] == pytest.approx(0.5)
    assert scores["f1"] == pytest.approx(0.2)


def test_interpolating_function_with_histograms(tmp_path):
    def fn(x):
        return np.asarray(x) * 2.0 + 1.0

    rep = _report(_base_df(400), tmp_path, ml_qual_to_qual_fn=fn)
    figs = _capture_figs(rep)
    rep.plot_interpolating_function_with_histograms(output_filename="cal")
    fig = figs[0]
    assert len(fig.axes) == 4
    deriv_ax = fig.axes[2]
    # derivative axis: 0.1 / (fn(x + 0.1) - fn(x)) == 0.5 everywhere for a slope-2 link
    np.testing.assert_allclose(deriv_ax.get_lines()[0].get_ydata(), 0.5)


def test_quality_histogram_mixed_with_interpolating_function(tmp_path):
    def fn(x):
        return np.asarray(x) + 10.0

    rep = _report(_add_mixed(_base_df(800)), tmp_path, mode="mixed", ml_qual_to_qual_fn=fn)
    figs = _capture_figs(rep)
    rep.plot_quality_histogram(plot_interpolating_function=True, output_filename="qh")
    assert len(figs[0].axes) == 3
    legacy = pd.read_hdf(rep.output_h5_filename, "quality_histogram")
    names = set(legacy.columns.get_level_values(0))
    assert names == {"is_mixed non-mixed", "is_mixed mixed, exactly one end", "is_mixed mixed, both ends"}
    display = pd.read_hdf(rep.output_h5_filename, "quality_histogram_mixed_start")
    assert set(display.columns.get_level_values(0)) == {"is_mixed_start non-mixed", "is_mixed_start mixed"}


def test_numerical_feature_hist_and_qual(tmp_path):
    fm = _add_pe_duplex(_base_df(800))
    fm["all_nan"] = np.nan
    fm["const"] = 3.0
    rep = _report(fm, tmp_path)
    figs = _capture_figs(rep)
    rep.plot_numerical_feature_hist_and_qual("all_nan", output_filename="a")
    assert figs == []  # all-NaN column skipped
    stats = rep._get_stats_for_feature_plot("RL")
    assert stats["count"].sum() == len(fm)
    assert set(stats["read_group"].astype(str)) <= set(PE_DUPLEX_GROUPS)
    rep.plot_numerical_feature_hist_and_qual("BCSQ", nbins=10, output_filename="b")  # continuous -> binned
    rep.plot_numerical_feature_hist_and_qual("const", output_filename="c")  # constant -> discrete
    rep.plot_numerical_feature_hist_and_qual("REF", output_filename="d")  # object -> discrete
    assert len(figs) == 3
    # the quality panel (axes[0]) draws one median line per read group with > 50 TP rows at a feature value
    d = rep.data_df
    n_tp_per_group = d[d["label"]].groupby("read_group", observed=True).size()
    assert len(figs[1].axes[0].get_lines()) == int((n_tp_per_group > 50).sum()) > 0
    # two figure-level legends (histogram + per-group quality)
    assert len(figs[0].legends) == 2


def test_trinuc_plot_ngroup_split_snv_only(tmp_path):
    fm = _add_indels(_add_pe_duplex(_base_df(1200)), hmer=True, non_hmer=False)
    rep = _report(fm, tmp_path)
    _capture_figs(rep)
    rep.calc_and_plot_trinuc_plot(output_filename="tri")
    stats = pd.read_hdf(rep.output_h5_filename, "trinuc_stats")
    present = {g for g in PE_DUPLEX_GROUPS if ((rep.data_df["pe_duplex_group"] == g) & rep.data_df["label"]).any()}
    for g in present:
        assert f"mixed={g} median_qual" in stats.columns
    # histogram counts: label columns are "<label> (<count>)" with count == SNV rows of that label only
    snv = rep.data_df[rep.data_df["variant_type"] == "snv"]
    count_cols = [c for c in stats.columns if c.startswith("True (")]
    assert count_cols == [f"True ({int(snv['label'].sum())})"]


def test_logit_histograms_split_by_variant_type(tmp_path):
    fm = _add_indels(_add_pe_duplex(_base_df(1200)))
    rep = _report(fm, tmp_path)
    figs = _capture_figs(rep)
    rep.plot_logit_histograms(
        output_filename="snv", output_filename_hmer_indel="h", output_filename_non_hmer_indel="nh", plot_by_fold=True
    )
    assert len(figs) == 3
    keys = _h5_keys(rep)
    assert {"/logit_histogram", "/logit_histogram_pe_duplex"} <= keys
    hist = pd.read_hdf(rep.output_h5_filename, "logit_histogram_pe_duplex")
    labels = {c.strip() for c in hist.columns.get_level_values(0)}
    assert "FP" in labels


def test_logit_histograms_skip_non_hmer_without_rows(tmp_path):
    fm = _add_indels(_add_pe_duplex(_base_df(800)), non_hmer=False)
    rep = _report(fm, tmp_path)
    figs = _capture_figs(rep)
    rep.plot_logit_histograms(output_filename="snv", output_filename_non_hmer_indel="nh", plot_by_fold=False)
    assert len(figs) == 1


# ───────────────────────────── hmer-indel sections ─────────────────────────────


def test_hmer_indel_by_class_skips_and_single_group(tmp_path):
    rep = _report(_base_df(300), tmp_path)
    figs = _capture_figs(rep)
    rep.plot_hmer_indel_by_class(output_filename="x")  # no variant_type / X_IC / X_HMER_RUN
    df_fp_only = _add_indels(_base_df(300), non_hmer=False)
    df_fp_only.loc[df_fp_only["variant_type"] == "hmer_indel", "label"] = False
    rep2 = _report(df_fp_only, tmp_path / "b")
    rep2._save_plt = rep._save_plt
    rep2.plot_hmer_indel_by_class(output_filename="x")  # no hmer TP rows
    assert figs == []

    # none-mode (single "all reads" group) -> lines split only by ins/del, read_group == "all"
    fm = _add_indels(_base_df(800), non_hmer=False)
    rep3 = _report(fm, tmp_path / "c")
    figs3 = _capture_figs(rep3)
    rep3.plot_hmer_indel_by_class(output_filename="x")
    stats = pd.read_hdf(rep3.output_h5_filename, "hmer_indel_class_stats")
    assert set(stats["read_group"]) == {"all"}
    assert stats["hmer_length"].max() <= 12
    ind = rep3.data_df[(rep3.data_df["variant_type"] == "hmer_indel") & rep3.data_df["label"]]
    assert stats["count"].sum() == len(ind)
    n_cls = stats["indel_class"].nunique()
    assert len(figs3[0].axes[0].get_lines()) == n_cls
    assert {t.get_text() for t in figs3[0].axes[0].get_legend().get_texts()} <= {"insertion", "deletion"}


def test_hmer_indel_by_class_object_read_group(tmp_path):
    """A non-categorical (object) read_group with >1 present group splits the lines per group."""
    fm = _add_indels(_add_pe_duplex(_base_df(800)), non_hmer=False)
    rep = _report(fm, tmp_path)
    rep.data_df["read_group"] = rep.data_df["read_group"].astype(str)
    _capture_figs(rep)
    rep.plot_hmer_indel_by_class()
    stats = pd.read_hdf(rep.output_h5_filename, "hmer_indel_class_stats")
    assert len(set(stats["read_group"])) > 1


def test_calc_hmer_indel_auc_table_values(tmp_path):
    from sklearn.metrics import roc_auc_score  # noqa: PLC0415

    fm = _add_indels(_add_pe_duplex(_base_df(1500)))
    # make non-hmer deletions all TP -> AUC undefined for that group
    nh_del = (fm["variant_type"] == "non_hmer_indel") & (fm["X_IC"] == "del")
    fm.loc[nh_del, "label"] = True
    rep = _report(fm, tmp_path)
    rep.calc_hmer_indel_auc_table()
    table = pd.read_hdf(rep.output_h5_filename, "hmer_indel_auc_table").set_index("group")
    assert list(table.index) == [
        "snv",
        "hmer_indel",
        "hmer_indel:ins",
        "hmer_indel:del",
        "non_hmer_indel",
        "non_hmer_indel:ins",
        "non_hmer_indel:del",
    ]
    d = rep.data_df
    hm = d[d["variant_type"] == "hmer_indel"]
    assert table.loc["hmer_indel", "n_TP"] == hm["label"].sum()
    assert table.loc["hmer_indel", "n_FP"] == (~hm["label"]).sum()
    auc = roc_auc_score(hm["label"].astype(int), hm["MQUAL"])
    assert table.loc["hmer_indel", "roc_auc"] == pytest.approx(auc)
    assert table.loc["hmer_indel", "roc_auc_phred"] == pytest.approx(-10 * np.log10(1 - auc))
    assert table.loc["non_hmer_indel:del", "n_FP"] == 0
    assert np.isnan(table.loc["non_hmer_indel:del", "roc_auc"])


def test_calc_hmer_indel_auc_table_skips_snv_only(tmp_path):
    rep = _report(_base_df(300), tmp_path)
    rep.calc_hmer_indel_auc_table()
    fm = _base_df(300)
    fm["variant_type"] = "snv"
    rep2 = _report(fm, tmp_path / "b")
    rep2.calc_hmer_indel_auc_table()
    assert "/hmer_indel_auc_table" not in _h5_keys(rep) | _h5_keys(rep2)


def test_hmer_indel_class_conditions(tmp_path):
    rep = _report(_base_df(200), tmp_path)
    assert rep._hmer_indel_class_conditions() is None  # no variant_type column
    fm = _base_df(200)
    fm["variant_type"] = "snv"
    assert _report(fm, tmp_path / "b")._hmer_indel_class_conditions() is None  # no indel rows
    df_nh = _add_indels(_base_df(600), hmer=False).drop(columns=["X_HMER_RUN"])
    conds = _report(df_nh, tmp_path / "c")._hmer_indel_class_conditions()
    assert list(conds) == ["snv (baseline)", "non-hmer indel", "non-hmer ins", "non-hmer del"]
    vt = df_nh["variant_type"].to_numpy()
    np.testing.assert_array_equal(conds["non-hmer indel"], vt == "non_hmer_indel")
    np.testing.assert_array_equal(conds["non-hmer ins"] | conds["non-hmer del"], conds["non-hmer indel"])
    # hmer rows without X_IC -> class masks fall back to empty strings (no ins/del rows)
    df_h = _add_indels(_base_df(400), non_hmer=False).drop(columns=["X_IC"])
    conds_h = _report(df_h, tmp_path / "d")._hmer_indel_class_conditions()
    assert not conds_h["ins"].any() and not conds_h["del"].any()
    run = df_h["X_HMER_RUN"].to_numpy()
    is_h = df_h["variant_type"].to_numpy() == "hmer_indel"
    np.testing.assert_array_equal(conds_h["hmer run 13+"], is_h & (run >= 13))


def test_hmer_indel_run_info_table_values(tmp_path):
    fm = _add_indels(_add_pe_duplex(_base_df(1500)))
    rep = _report(fm, tmp_path)
    rep.calc_hmer_indel_run_info_table()
    t = pd.read_hdf(rep.output_h5_filename, "run_quality_summary_table_hmer_indel")
    assert {"non-hmer indel", "non-hmer ins", "non-hmer del", "hmer run 13+"} <= set(t["indel_class"])
    assert set(t["read_type"]) == {"All reads", *PE_DUPLEX_GROUPS}
    d = rep.data_df
    sel = (d["variant_type"] == "non_hmer_indel") & (d["pe_duplex_group"] == "duplex PE")
    row = t[(t["indel_class"] == "non-hmer indel") & (t["read_type"] == "duplex PE")].iloc[0]
    tp = d[sel & d["label"]]
    assert row["n_TP"] == len(tp) and row["n_FP"] == (sel & ~d["label"]).sum()
    assert row["Median SNVQ"] == pytest.approx(spu.signif(tp["SNVQ"].median(), spu.SIG_DIGITS))
    assert row["Recall at SNVQ=50"] == pytest.approx(spu.signif((tp["SNVQ"] >= 50).mean(), spu.SIG_DIGITS))
    # cross-product size: classes x (All reads + 4 groups)
    assert len(t) == t["indel_class"].nunique() * 5


def test_pe_concordance_summary_values_and_skips(tmp_path):
    rep = _report(_add_indels(_base_df(800)), tmp_path)  # no CS / pe-duplex columns
    rep.calc_pe_concordance_summary_table()
    rep_snv = _report(_add_pe_duplex(_base_df(300)), tmp_path / "b")  # pe-duplex but no indel rows
    rep_snv.calc_pe_concordance_summary_table()
    assert "/run_quality_summary_table_pe_concordance" not in _h5_keys(rep) | _h5_keys(rep_snv)

    fm = _add_indels(_add_pe_duplex(_base_df(2000)))
    rep2 = _report(fm, tmp_path / "c")
    d = rep2.data_df
    # force one cell (non-hmer del, concordant) to contain only TPs -> AUC NaN
    pe = d["pe_duplex_group"].isin(["simplex PE", "duplex PE"])
    conc = pe & (d["cs_n_crossing"] == d["cs_n_supporting"])
    cell = conc & (d["variant_type"] == "non_hmer_indel") & (d["X_IC"] == "del")
    rep2.data_df.loc[cell, "label"] = True
    rep2.calc_pe_concordance_summary_table()
    t = pd.read_hdf(rep2.output_h5_filename, "run_quality_summary_table_pe_concordance")
    assert set(t["concordance"]) == {"concordant", "non-concordant"}
    d = rep2.data_df
    hm_conc = conc & (d["variant_type"] == "hmer_indel")
    row = t[(t["indel_class"] == "all indel") & (t["concordance"] == "concordant")].iloc[0]
    assert row["n_TP"] == (hm_conc & d["label"]).sum()
    assert row["n_FP"] == (hm_conc & ~d["label"]).sum()
    tp_snvq = d.loc[hm_conc & d["label"], "SNVQ"]
    assert row["TP%>=60"] == pytest.approx(spu.signif(100 * (tp_snvq >= 60).mean(), spu.SIG_DIGITS))
    nh = t[(t["indel_class"] == "non-hmer del") & (t["concordance"] == "concordant")].iloc[0]
    assert nh["n_FP"] == 0 and np.isnan(nh["ROC AUC (Phred)"]) and np.isnan(nh["FP%>=60"])


def test_hmer_indel_fq_recall(tmp_path):
    fm = _add_indels(_add_pe_duplex(_base_df(1500)), non_hmer=False)
    rep = _report(fm, tmp_path)
    figs = _capture_figs(rep)
    rep.plot_hmer_indel_fq_recall(output_filename="x")  # base_recall not set -> skip
    assert figs == []
    rep.plot_fq_recall(only_calculate=True)
    rep.plot_hmer_indel_fq_recall(output_filename="x")
    fq = pd.read_hdf(rep.output_h5_filename, "FQ_recall_LoD_hmer_indel")
    for lbl in ("all indel", "ins", "del"):
        assert f"recall_{lbl}" in fq.columns and f"FQ_{lbl}" in fq.columns
    # the conditional recall of a sub-class never exceeds the all-indel recall
    assert (fq["recall_ins"] <= fq["recall_all indel"] + 1e-12).all()
    assert len(figs[0].axes[0].get_lines()) == 3

    rep_snv = _report(_base_df(300), tmp_path / "b")
    rep_snv.base_recall = 0.5
    rep_snv.plot_hmer_indel_fq_recall()
    assert "/FQ_recall_LoD_hmer_indel" not in _h5_keys(rep_snv)


def test_hmer_indel_fq_recall_skips_empty_class(tmp_path):
    fm = _add_indels(_add_pe_duplex(_base_df(1500)), non_hmer=False)
    fm.loc[fm["variant_type"] == "hmer_indel", "X_IC"] = "ins"  # no deletions
    rep = _report(fm, tmp_path)
    figs = _capture_figs(rep)
    rep.plot_fq_recall(only_calculate=True)
    rep.plot_hmer_indel_fq_recall()
    fq = pd.read_hdf(rep.output_h5_filename, "FQ_recall_LoD_hmer_indel")
    assert "recall_del" not in fq.columns
    assert len(figs[-1].axes[0].get_lines()) == 2


def test_hmer_indel_context_df_none_cases(tmp_path):
    assert _report(_base_df(200), tmp_path)._hmer_indel_context_df() is None  # SNV-only
    fm = _add_indels(_base_df(400), non_hmer=False).drop(columns=["X_HMER_PRE"])
    assert _report(fm, tmp_path / "b")._hmer_indel_context_df() is None  # missing snvfind column
    df2 = _add_indels(_base_df(400), non_hmer=False)
    df2["X_HMER_BASE"] = "N"  # no valid context rows
    assert _report(df2, tmp_path / "c")._hmer_indel_context_df() is None


def test_hmer_indel_context_df_counts(tmp_path):
    fm = _add_indels(_add_pe_duplex(_base_df(1500)), non_hmer=False)
    rep = _report(fm, tmp_path)
    table = rep._hmer_indel_context_df()
    d = rep.data_df
    hm = d[d["variant_type"] == "hmer_indel"]
    assert table["n_TP"].sum() == hm["label"].sum()
    assert table["n_FP"].sum() == (~hm["label"]).sum()
    assert table["hmer_len"].between(1, 10).all()
    assert set(table["read_group"]) <= set(PE_DUPLEX_GROUPS)


def test_hmer_indel_context_plot_layout_and_ylim(tmp_path):
    fm = _add_indels(_add_pe_duplex(_base_df(1500)), non_hmer=False)
    # some rows whose flanking base equals the hmer base -> outside the context axis (skipped)
    hm_idx = fm.index[fm["variant_type"] == "hmer_indel"][:20]
    fm.loc[hm_idx, "X_HMER_PRE"] = fm.loc[hm_idx, "X_HMER_BASE"]
    rep = _report(fm, tmp_path)
    figs = _capture_figs(rep)
    rep.calc_and_plot_hmer_indel_context_plot(output_filename="ctx")
    fig = figs[0]
    assert len(fig.axes) == 4 * 2 * 2  # 4 hmer bases x (ins, del) x (quality, density)
    table = pd.read_hdf(rep.output_h5_filename, "hmer_indel_context_stats")
    med = table["median_snvq_tp"].dropna()
    lo, hi = fig.axes[0].get_ylim()
    assert lo <= med.min() and hi >= med.max()
    assert len(fig.axes[1].get_xticklabels()) == 3 * 3 * 10


def test_hmer_indel_context_plot_ylim_fallbacks(tmp_path):
    # all-FP hmer indels -> no TP SNVQ -> default 40..100 SNVQ range
    fm = _add_indels(_add_pe_duplex(_base_df(800)), non_hmer=False)
    fm.loc[fm["variant_type"] == "hmer_indel", "label"] = False
    rep = _report(fm, tmp_path)
    figs = _capture_figs(rep)
    rep.calc_and_plot_hmer_indel_context_plot()
    assert figs[0].axes[0].get_ylim() == (40.0, 100.0)
    # constant TP SNVQ -> near-flat range widened to the minimum 10-unit span
    df2 = _add_indels(_add_pe_duplex(_base_df(800)), non_hmer=False)
    df2.loc[df2["variant_type"] == "hmer_indel", "SNVQ"] = 70.0
    rep2 = _report(df2, tmp_path / "b")
    figs2 = _capture_figs(rep2)
    rep2.calc_and_plot_hmer_indel_context_plot()
    lo, hi = figs2[0].axes[0].get_ylim()
    assert hi == pytest.approx(73.0) and hi - lo == pytest.approx(10.0)


def test_hmer_indel_context_plot_skips_snv_only(tmp_path):
    rep = _report(_base_df(200), tmp_path)
    figs = _capture_figs(rep)
    rep.calc_and_plot_hmer_indel_context_plot()
    assert figs == [] and "/hmer_indel_context_stats" not in _h5_keys(rep)


# ───────────────────────────── non-hmer indel sections ─────────────────────────────


def test_non_hmer_by_class_single_group_and_skips(tmp_path):
    rep = _report(_base_df(200), tmp_path)
    figs = _capture_figs(rep)
    rep.plot_non_hmer_indel_by_class()  # missing columns
    df_fp = _add_indels(_base_df(600), hmer=False)
    df_fp.loc[df_fp["variant_type"] == "non_hmer_indel", "label"] = False
    rep_fp = _report(df_fp, tmp_path / "b")
    rep_fp._save_plt = rep._save_plt
    rep_fp.plot_non_hmer_indel_by_class()  # no TP rows
    assert figs == []

    fm = _add_indels(_base_df(800), hmer=False)
    rep2 = _report(fm, tmp_path / "c")
    figs2 = _capture_figs(rep2)
    rep2.plot_non_hmer_indel_by_class()
    stats = pd.read_hdf(rep2.output_h5_filename, "non_hmer_indel_class_stats")
    assert set(stats["read_group"]) == {"all"}
    # |X_IL| clipped to [1, 20]: X_IL==0 is clipped up to 1
    assert stats["indel_length"].between(1, 20).all()
    tp = rep2.data_df[(rep2.data_df["variant_type"] == "non_hmer_indel") & rep2.data_df["label"]]
    assert stats["count"].sum() == len(tp)
    assert len(figs2[0].axes[0].get_lines()) == stats["indel_class"].nunique()


def test_indel_read_group_split_drops_ungrouped_rows(tmp_path):
    rep = _report(_add_pe_duplex(_base_df(200)), tmp_path)
    ind = pd.DataFrame({"read_group": pd.Categorical(["a", "b", None, "a"], categories=["a", "b"])})
    base = pd.DataFrame({"x": [1, 2, 3, 4]})
    groups, out = rep._indel_read_group_split(ind, base)
    assert groups == ["a", "b"]
    assert list(out["read_group"]) == ["a", "b", "a"]
    groups1, out1 = rep._indel_read_group_split(ind.iloc[[0, 3]], base.iloc[[0, 3]])
    assert groups1 == ["all"] and list(out1["read_group"]) == ["all", "all"]
    groups2, _ = rep._indel_read_group_split(pd.DataFrame(index=base.index), base)
    assert groups2 == ["all"]


def test_non_hmer_str_summary(tmp_path):
    fm = _add_indels(_add_pe_duplex(_base_df(1500)), hmer=False)
    rep = _report(fm, tmp_path)
    figs = _capture_figs(rep)
    rep.plot_non_hmer_indel_str()
    stats = pd.read_hdf(rep.output_h5_filename, "non_hmer_indel_str_stats")
    d = rep.data_df
    is_str = (d["variant_type"] == "non_hmer_indel") & d["label"] & (d["RU"] != ".")
    for axis in ("period", "copies"):
        sub = stats[stats["axis"] == axis]
        assert sub["count"].sum() == is_str.sum()
        assert sub["value"].between(1, 6).all()
    assert len(figs[0].axes) == 4

    rep_none = _report(_add_indels(_base_df(300), hmer=False).drop(columns=["RPA"]), tmp_path / "b")
    rep_none.plot_non_hmer_indel_str()
    df_no_str = _add_indels(_base_df(300), hmer=False)
    df_no_str["RU"] = "."
    rep_ns = _report(df_no_str, tmp_path / "c")
    rep_ns.plot_non_hmer_indel_str()
    assert "/non_hmer_indel_str_stats" not in _h5_keys(rep_none) | _h5_keys(rep_ns)


def test_non_hmer_str_context_df_none_cases(tmp_path):
    assert _report(_base_df(200), tmp_path)._non_hmer_str_context_df() is None  # no non-hmer rows
    fm = _add_indels(_base_df(400), hmer=False).drop(columns=["RPA"])
    assert _report(fm, tmp_path / "b")._non_hmer_str_context_df() is None  # missing RPA
    df2 = _add_indels(_base_df(400), hmer=False)
    df2["RU"] = "."
    assert _report(df2, tmp_path / "c")._non_hmer_str_context_df() is None  # no repeat units
    df3 = _add_indels(_base_df(400), hmer=False)
    df3.loc[df3["RU"] != ".", "RU"] = "AAAAT"  # period 5 only -> outside di/tri/tetra
    assert _report(df3, tmp_path / "d")._non_hmer_str_context_df() is None


def test_non_hmer_str_context_canonical_motifs(tmp_path):
    fm = _add_indels(_add_pe_duplex(_base_df(1000)), hmer=False)
    rep = _report(fm, tmp_path)
    table = rep._non_hmer_str_context_df()
    assert set(table["period"]) <= {2, 3, 4}
    # TTA canonicalizes to AAT; CAG to AGC; motifs are canonical (rotation/revcomp-min)
    assert all(m == SRSNVReport._canonical_repeat_unit(m) for m in table["motif"])
    assert "TTA" not in set(table["motif"])
    d = rep.data_df
    sel = (d["variant_type"] == "non_hmer_indel") & d["RU"].isin(["AC", "AG", "CAG", "TTA", "ACGT"])
    assert (table["n_TP"] + table["n_FP"]).sum() == sel.sum()


def _many_motif_df(n=2000, seed=9):
    """Non-hmer indels over >10 canonical tetranucleotide motifs (exceeds the top-K x-axis; note there are
    exactly 10 canonical non-homopolymer trinucleotide motifs, so tetranucleotides are needed)."""
    fm = _add_indels(_add_pe_duplex(_base_df(n, seed=seed)), hmer=False, n_non_hmer=1200)
    tri = [a + b + c + e for a in "ACGT" for b in "ACGT" for c in "ACGT" for e in "ACGT" if len({a, b, c, e}) > 2]
    rng = np.random.default_rng(seed)
    nh = fm["variant_type"] == "non_hmer_indel"
    fm.loc[nh, "RU"] = rng.choice(tri, nh.sum())
    fm.loc[nh, "RPA"] = rng.integers(1, 6, nh.sum())
    return fm


def test_non_hmer_str_context_plot_topk_and_layout(tmp_path):
    rep = _report(_many_motif_df(), tmp_path)
    figs = _capture_figs(rep)
    rep.calc_and_plot_non_hmer_str_context_plot(output_filename="ctx")
    table = pd.read_hdf(rep.output_h5_filename, "non_hmer_indel_str_context_stats")
    assert table["motif"].nunique() > SRSNVReport._STR_CTX_TOPK
    fig = figs[0]
    assert len(fig.axes) == 4  # one period band x (ins, del) x (quality, density)
    assert len(fig.axes[1].get_xticklabels()) == SRSNVReport._STR_CTX_TOPK
    # density panel normalized within the top-K motifs
    heights = [p.get_height() for p in fig.axes[1].patches[: SRSNVReport._STR_CTX_TOPK]]
    assert sum(heights) == pytest.approx(1.0)


def test_non_hmer_str_context_plot_ylim_fallbacks(tmp_path):
    fm = _many_motif_df()
    fm.loc[fm["variant_type"] == "non_hmer_indel", "label"] = False
    rep = _report(fm, tmp_path)
    figs = _capture_figs(rep)
    rep.calc_and_plot_non_hmer_str_context_plot()
    assert figs[0].axes[0].get_ylim() == (40.0, 100.0)

    df2 = _many_motif_df()
    df2.loc[df2["variant_type"] == "non_hmer_indel", "SNVQ"] = 50.0
    rep2 = _report(df2, tmp_path / "b")
    figs2 = _capture_figs(rep2)
    rep2.calc_and_plot_non_hmer_str_context_plot()
    lo, hi = figs2[0].axes[0].get_ylim()
    assert hi == pytest.approx(53.0) and hi - lo == pytest.approx(10.0)


def test_non_hmer_str_context_plot_skips(tmp_path):
    rep = _report(_base_df(200), tmp_path)
    figs = _capture_figs(rep)
    rep.calc_and_plot_non_hmer_str_context_plot()
    assert figs == []
    # defensive guard: a context table with no di/tri/tetra period renders nothing
    rep2 = _report(_add_indels(_base_df(400), hmer=False), tmp_path / "b")
    figs2 = _capture_figs(rep2)
    fake = pd.DataFrame(
        {"period": [5], "motif": ["AAAAT"], "ins_del": ["ins"], "read_group": ["all reads"]}
        | {"n_TP": [1], "median_snvq_tp": [50.0], "n_FP": [0]}
    )
    with patch.object(SRSNVReport, "_non_hmer_str_context_df", return_value=fake):
        rep2.calc_and_plot_non_hmer_str_context_plot()
    assert figs2 == []


# ───────────────────────────── end-to-end create_report ─────────────────────────────


def test_create_report_end_to_end_pe_duplex_with_indels(tmp_path):
    fm = _add_indels(_add_pe_duplex(_base_df(1500)))
    models = [_ModelWithTrainingResults(m) for m in _dummy_models(2)]
    rep = _report(fm, tmp_path, models=models, ml_qual_to_qual_fn=lambda x: np.asarray(x) * 1.5 + 20)
    figs = _capture_figs(rep)
    rep.create_report()
    keys = _h5_keys(rep)
    expected = {
        "/FQ_recall_LoD",
        "/FQ_recall_LoD_pe_duplex",
        "/FQ_recall_LoD_hmer_indel",
        "/run_info_table",
        "/run_quality_summary_table",
        "/run_quality_table",
        "/ppmseq_category_quality_table",
        "/roc_auc_table",
        "/logit_histogram",
        "/quality_histogram",
        "/trinuc_stats",
        "/hmer_indel_class_stats",
        "/hmer_indel_auc_table",
        "/run_quality_summary_table_hmer_indel",
        "/run_quality_summary_table_pe_concordance",
        "/hmer_indel_context_stats",
        "/non_hmer_indel_class_stats",
        "/non_hmer_indel_str_stats",
        "/non_hmer_indel_str_context_stats",
        "/keys_to_convert",
    }
    assert expected <= keys, expected - keys
    assert "/training_progress" not in keys and "/mean_abs_SHAP_scores" not in keys  # dummy models
    ktc = pd.read_hdf(rep.output_h5_filename, "keys_to_convert")
    assert "trinuc_stats" in set(ktc) and "FQ_recall_LoD" in set(ktc)
    assert len(figs) >= 15


# ───────────────────────────── remaining PR-changed branches ─────────────────────────────


def test_hmer_indel_by_class_without_read_group_column(tmp_path):
    rep = _report(_add_indels(_add_pe_duplex(_base_df(800)), non_hmer=False), tmp_path)
    rep.data_df = rep.data_df.drop(columns=["read_group"])
    _capture_figs(rep)
    rep.plot_hmer_indel_by_class()
    stats = pd.read_hdf(rep.output_h5_filename, "hmer_indel_class_stats")
    assert set(stats["read_group"]) == {"all"}


def test_hmer_indel_class_conditions_without_run_length(tmp_path):
    fm = _add_indels(_base_df(600), non_hmer=False).drop(columns=["X_HMER_RUN"])
    conds = _report(fm, tmp_path)._hmer_indel_class_conditions()
    assert list(conds) == ["snv (baseline)", "all indel", "ins", "del"]


def test_hmer_indel_context_plot_sparse_groups(tmp_path):
    """Only two hmer rows: one in-axis (duplex PE, hmer A, ins) and one whose flank equals the hmer base
    (simplex SE) -> only one SNVQ step-line is drawn on the A/ins quality panel and none elsewhere."""
    fm = _add_pe_duplex(_base_df(400))
    rep = _report(fm, tmp_path)
    d = rep.data_df
    d["variant_type"] = "snv"
    d["X_IC"] = None
    i0, i1 = d.index[0], d.index[1]
    d.loc[[i0, i1], "variant_type"] = "hmer_indel"
    d.loc[[i0, i1], "X_IC"] = "ins"
    d.loc[[i0, i1], "label"] = True
    d["X_HMER_BASE"], d["X_HMER_POST"], d["X_HMER_RUN"] = "A", "C", 3
    d["X_HMER_PRE"] = "G"
    d.loc[i1, "X_HMER_PRE"] = "A"  # flank == hmer base -> outside the context axis
    d["pe_duplex_group"] = d["pe_duplex_group"].astype(object)
    d.loc[i0, "pe_duplex_group"] = "duplex PE"
    d.loc[i1, "pe_duplex_group"] = "simplex SE"
    figs = _capture_figs(rep)
    rep.calc_and_plot_hmer_indel_context_plot()
    table = pd.read_hdf(rep.output_h5_filename, "hmer_indel_context_stats")
    assert set(table["read_group"]) == {"duplex PE", "simplex SE"}
    axes = figs[0].axes
    a_ins_quality = axes[0]
    # duplex PE step line + SNVQ=60 line + 9 separators; the out-of-axis simplex SE row draws nothing
    assert len(a_ins_quality.get_lines()) == 1 + 1 + 9
    c_ins_quality = axes[4]  # band 2 (hmer C) has no rows for any group -> no step lines
    assert len(c_ins_quality.get_lines()) == 1 + 9


def test_non_hmer_indel_by_class_keeps_lengths_up_to_20(tmp_path):
    fm = _add_indels(_base_df(800), hmer=False)
    nh = fm.index[(fm["variant_type"] == "non_hmer_indel") & fm["label"].astype(bool)]
    assert len(nh) >= 3
    fm.loc[nh[0], "X_IL"] = 15
    fm.loc[nh[1], "X_IL"] = -20
    fm.loc[nh[2], "X_IL"] = 27  # beyond the synthesis maximum: lumped into the 20 bin
    rep = _report(fm, tmp_path)
    _capture_figs(rep)
    rep.plot_non_hmer_indel_by_class()
    stats = pd.read_hdf(rep.output_h5_filename, "non_hmer_indel_class_stats")
    lengths = set(stats["indel_length"])
    assert {15, 20} <= lengths
    assert stats["indel_length"].max() == 20
