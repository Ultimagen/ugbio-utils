import numpy as np
import pandas as pd
import pytest
from ugbio_filtering import transformers
from ugbio_filtering.tprep_constants import GtType, VcfType
from ugbio_filtering.variant_filtering_utils import apply_model, train_model


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
            "segdup": 0.1,
            "uniq_frac": 0.95,
            "low_conf": None,
            "cnv_source": ("depth",),
            "sr_reads": None,
            "snap": "SR",
            "snap_n": 9,
            "depth_pos": 1001,
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
            "segdup": 0.9,
            "uniq_frac": None,
            "low_conf": True,
            "cnv_source": ("split_read",),
            "sr_reads": 4,
            "snap": "none",
            "snap_n": 0,
            "depth_pos": 501,
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


class TestDepthCnvFeatures:
    def test_features(self):
        f = transformers.depth_cnv_features(depth_cnv_df())
        assert list(f.columns) == transformers.DEPTH_CNV_FEATURES
        d, u = f.iloc[0], f.iloc[1]
        assert (d.is_del, u.is_del) == (1.0, 0.0)
        assert d.log_len == pytest.approx(np.log10(5000))
        assert (d.dcn, d.rel_cn) == (pytest.approx(1.0), pytest.approx(0.5))
        assert (u.dcn, u.rel_cn) == (pytest.approx(1.0), pytest.approx(2.0))  # haploid contig: baseline 1
        # support is taken for the call's own type
        assert (d.sr_same, d.sr_opp, d.pe_same, d.pe_opp) == (12, 1, 7, 0)
        assert (u.sr_same, u.sr_opp, u.pe_same, u.pe_opp) == (3, 0, 5, 2)
        assert (d.sr_insert, u.sr_insert, d.pe_span, u.pe_span) == (5000, 1900, 5400, 2100)
        assert d.log_support == pytest.approx(np.log1p(19))
        assert (d.lowconf, u.lowconf) == (0.0, 1.0)
        assert (d.snap_sr, d.snap_pe, u.snap_sr) == (1.0, 0.0, 0.0)
        assert d.snap_shrink == pytest.approx(1 - 5000 / 6000)
        assert (d.source_sr, u.source_sr, u.sr_reads, d.sr_reads) == (0.0, 1.0, 4, 0)
        assert (d.source_pair, u.source_pair) == (0.0, 0.0)
        assert u.uniq_frac == 0.0  # missing -> 0
        assert not f.isna().any().any()

    def test_pair_source(self):
        calls = depth_cnv_df(2)
        calls["cnv_source"] = [("pair",), ("depth", "split_read")]
        f = transformers.depth_cnv_features(calls)
        assert list(f.source_pair) == [1.0, 0.0]
        assert list(f.source_sr) == [0.0, 0.0]  # only calls found by split reads alone

    def test_cnmops_source_is_opt_in(self):
        calls = depth_cnv_df(2)
        calls["cnv_source"] = [("cn.mops",), ("depth",)]
        assert list(transformers.depth_cnv_features(calls).columns) == transformers.DEPTH_CNV_FEATURES
        f = transformers.depth_cnv_features(calls, cnmops=True)
        assert list(f.columns) == transformers.DEPTH_CNV_FEATURES_CNMOPS
        assert list(f.source_cnmops) == [1.0, 0.0]


class TestDepthCnvModel:
    def test_transformer_train_and_apply(self):
        calls = depth_cnv_df(40)
        calls["label"] = [1, 0] * 20
        model, fitted = train_model(calls, GtType.APPROXIMATE, VcfType.CNV_DEPTH)  # returns (model, transformer)
        assert model.n_estimators == 500
        x = fitted.transform(calls)
        assert list(x.columns) == [f"depth_cnv__{f}" for f in transformers.DEPTH_CNV_FEATURES]
        predictions, probabilities = apply_model(calls, model, fitted)
        assert list(predictions) == [1, 0] * 20
        assert probabilities.shape == (40, 2)

    def test_cnmops_model_has_the_extra_feature(self):
        calls = depth_cnv_df(40)
        calls["label"] = [1, 0] * 20
        _, fitted = train_model(calls, GtType.APPROXIMATE, VcfType.CNV_DEPTH_CNMOPS)
        assert list(fitted.get_feature_names_out()) == [
            f"depth_cnv__{f}" for f in transformers.DEPTH_CNV_FEATURES_CNMOPS
        ]

    def test_needed_features_are_the_info_columns(self):
        assert transformers.get_needed_features(VcfType.CNV_DEPTH) == [
            f.lower() for f in transformers.DEPTH_CNV_INFO_FIELDS
        ]
