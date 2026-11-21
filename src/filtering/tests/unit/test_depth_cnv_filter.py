import numpy as np
import pandas as pd
import pytest
from ugbio_filtering import transformers
from ugbio_filtering.tprep_constants import GtType, VcfType
from ugbio_filtering.variant_filtering_utils import apply_model, max_f1_decision_threshold, train_model


def depth_cnv_df(n: int = 2) -> pd.DataFrame:
    """get_vcf_df-style rows (lower-case INFO columns) for a deletion and a split-read duplication on chrX."""
    rows = [
        {
            "svtype": "DEL",
            "svlen": (5000,),
            "cn_mean": 1.0,
            "base_cn": 2,
            "z": -12.0,
            "n_merged": 2,
            "segdup_frac": 0.1,
            "uniq_frac": 0.95,
            "low_conf": None,
            "cnv_source": ("depth",),
            "cluster_support": None,
            "breakpoint_source": "split_read",
            "breakpoint_support": 9,
            "depth_start": 1001,
            "depth_end": 7001,
            "cnv_del_reads": 12,
            "cnv_dup_reads": 1,
            "cnv_total_reads": 600,
            "cnv_del_frac": 0.02,
            "cnv_dup_frac": 0.001,
            "cnv_del_pairs": 7,
            "cnv_dup_pairs": 0,
            "cnv_del_pairs_frac": 0.012,
            "cnv_dup_pairs_frac": 0.0,
            "del_reads_median_insert_size": 5000.0,
            "dup_reads_median_insert_size": None,
            "del_pairs_median_span": 5400.0,
            "dup_pairs_median_span": None,
        },
        {
            "svtype": "DUP",
            "svlen": (2000,),
            "cn_mean": 2.0,
            "base_cn": 1,
            "z": 6.0,
            "n_merged": 1,
            "segdup_frac": 0.9,
            "uniq_frac": None,
            "low_conf": True,
            "cnv_source": ("split_read",),
            "cluster_support": 4,
            "breakpoint_source": "depth",
            "breakpoint_support": 0,
            "depth_start": 501,
            "depth_end": 2501,
            "cnv_del_reads": 0,
            "cnv_dup_reads": 3,
            "cnv_total_reads": 200,
            "cnv_del_frac": 0.0,
            "cnv_dup_frac": 0.015,
            "cnv_del_pairs": 2,
            "cnv_dup_pairs": 5,
            "cnv_del_pairs_frac": 0.01,
            "cnv_dup_pairs_frac": 0.025,
            "del_reads_median_insert_size": None,
            "dup_reads_median_insert_size": 1900.0,
            "del_pairs_median_span": None,
            "dup_pairs_median_span": 2100.0,
        },
    ]
    return pd.DataFrame([rows[i % 2] for i in range(n)])


def cnv_features(calls: pd.DataFrame) -> pd.DataFrame:
    """The CNV model's features of the calls, named without the transformer prefix."""
    x = transformers.get_transformer(VcfType.CNV).fit_transform(calls)
    return x.rename(columns=lambda c: c.split("__", 1)[1])


class TestDepthCnvFeatures:
    def test_features(self):
        f = cnv_features(depth_cnv_df())
        assert list(f.columns) == [
            "svtype",
            "dcn",
            "svlen",
            *transformers.CNV_NUMERIC_FIELDS,
            "cnv_source",
            "breakpoint_source",
            "abs_z",
            "refine_shrink",
        ]
        d, u = f.iloc[0], f.iloc[1]
        assert (d.svtype, u.svtype) == (1.0, 2.0)  # svtype_encode: DEL 1, DUP 2
        assert (d.svlen, u.svlen) == (5000, 2000)
        assert (d.dcn, u.dcn) == (pytest.approx(1.0), pytest.approx(1.0))  # haploid contig: baseline 1
        assert (d.abs_z, u.abs_z) == (12.0, 6.0)
        assert (d.z, u.z) == (-12.0, 6.0)  # the INFO field as it is
        assert (d.cnv_del_reads, d.cnv_dup_reads, u.cnv_dup_pairs) == (12, 1, 5)
        assert (d.cnv_source, u.cnv_source) == (0.0, 1.0)  # depth, split_read
        assert (d.breakpoint_source, u.breakpoint_source) == (1.0, 0.0)  # split_read, depth
        assert d.refine_shrink == pytest.approx(1 - 5000 / 6000)
        assert u.refine_shrink == 0.0
        assert (u.uniq_frac, d.dup_reads_median_insert_size, u.cluster_support) == (0.0, 0.0, 4)  # missing -> 0
        assert not f.isna().any().any()

    def test_pair_source(self):
        calls = depth_cnv_df(2)
        calls["cnv_source"] = [("pair",), ("depth", "split_read")]
        assert list(cnv_features(calls).cnv_source) == [2.0, 0.0]  # the first source

    def test_missing_values_are_zero(self):
        calls = depth_cnv_df(2)
        calls["cnv_source"] = [np.nan, ("cn.mops",)]
        calls["breakpoint_source"] = [None, "pair"]
        calls["depth_start"] = [np.nan, 501]
        calls["depth_end"] = [np.nan, 2501]
        f = cnv_features(calls)
        assert list(f.cnv_source) == [0.0, 3.0]
        assert list(f.breakpoint_source) == [0.0, 2.0]
        assert f.refine_shrink.iloc[0] == 0.0  # no depth segment: nothing removed

    def test_region_tracks_are_not_features(self):
        f = cnv_features(depth_cnv_df())
        assert "segdup_frac" not in f and "low_conf" not in f


class TestDepthCnvModel:
    def test_transformer_train_and_apply(self):
        calls = depth_cnv_df(40)
        calls["label"] = [1, 0] * 20
        model, fitted = train_model(calls, GtType.APPROXIMATE, VcfType.CNV)  # returns (model, transformer)
        assert model.n_estimators == 500
        assert model.max_depth == 5
        x = fitted.transform(calls)
        assert x.shape == (40, 27)
        predictions, probabilities = apply_model(calls, model, fitted)
        assert list(predictions) == [1, 0] * 20
        assert probabilities.shape == (40, 2)

    def test_needed_features_are_the_info_columns(self):
        assert transformers.get_needed_features(VcfType.CNV) == [f.lower() for f in transformers.CNV_INFO_FIELDS]

    def test_plain_pickle_round_trip(self, tmp_path):
        # FilterVCF loads the model with plain pickle: the transformer must not hold local functions
        import pickle

        calls = depth_cnv_df(4)
        fitted = transformers.get_transformer(VcfType.CNV).fit(calls)
        path = tmp_path / "t.pkl"
        path.write_bytes(pickle.dumps(fitted))
        loaded = pickle.loads(path.read_bytes())  # noqa S301
        pd.testing.assert_frame_equal(loaded.transform(calls), fitted.transform(calls))


class TestDecisionThreshold:
    def test_max_f1_threshold(self):
        # true rows score 0.80 - 0.95 (36.0 - 42.8), false rows 0.10 - 0.75 (up to 34.8): 35 is the lowest clean cut
        scores = np.r_[np.linspace(0.8, 0.95, 20), np.linspace(0.1, 0.75, 20)]
        labels = np.r_[np.ones(20), np.zeros(20)]
        assert max_f1_decision_threshold(scores, labels) == 35

    def test_max_f1_threshold_takes_any_scores(self):
        # plain lists, no model; keeping the true row at 0.30 (26.3) is worth the false row at 0.35 (27.3), not the
        # one at 0.25 (25.2): 26, not 25, which would also pass 0.25 because FilterVCF passes a score above it
        scores = [0.25, 0.3, 0.35, 0.9, 0.9, 0.9, 0.1]
        labels = [0, 1, 0, 1, 1, 1, 0]
        assert max_f1_decision_threshold(scores, labels) == 26

    def test_given_threshold_wins(self):
        from ugbio_filtering.filter_variants_pipeline import resolve_decision_threshold

        assert resolve_decision_threshold(25.0, {"decision_threshold": 28}) == 25.0

    def test_stored_threshold_is_used(self):
        from ugbio_filtering.filter_variants_pipeline import resolve_decision_threshold

        assert resolve_decision_threshold(None, {"xgb": None, "decision_threshold": 28}) == 28.0

    def test_default_without_a_stored_threshold(self):
        from ugbio_filtering.filter_variants_pipeline import DEFAULT_DECISION_THRESHOLD, resolve_decision_threshold

        assert resolve_decision_threshold(None, {"xgb": None}) == DEFAULT_DECISION_THRESHOLD == 30.0
        assert resolve_decision_threshold(None, None) == 30.0
