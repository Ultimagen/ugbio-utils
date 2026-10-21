import os
from pathlib import Path

import pandas as pd
import pytest
from ugbio_filtering import train_models_pipeline

CNV_PAIR_FIELDS = [
    "cnv_dup_pairs",
    "cnv_del_pairs",
    "cnv_dup_pairs_frac",
    "cnv_del_pairs_frac",
    "dup_pairs_median_span",
    "del_pairs_median_span",
]


def _with_zero_pair_fields(src, dst):
    """Copy a CNV training dataframe, adding the read pair fields (written as 0 for single-end data)."""
    calls_df = pd.read_hdf(src, key="data")
    calls_df = calls_df.assign(**{c: 0 for c in CNV_PAIR_FIELDS})
    calls_df.to_hdf(dst, key="data")
    return dst


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


class TestRunTraining:
    def test_run_training_no_gt(self, tmpdir, resources_dir):
        train_models_pipeline.run(
            [
                "train_models_pipeline",
                "--train_dfs",
                f"{resources_dir}/train_model_approximate_gt_input.h5",
                "--test_dfs",
                f"{resources_dir}/train_model_approximate_gt_input.h5",
                "--output_file_prefix",
                f"{tmpdir}/approximate_gt.model",
                "--gt_type",
                "approximate",
                "--verbosity",
                "DEBUG",
            ]
        )
        assert os.path.exists(f"{tmpdir}/approximate_gt.model.h5")
        assert os.path.exists(f"{tmpdir}/approximate_gt.model.pkl")

    def test_run_training_cnv(self, tmpdir, resources_dir):
        train_models_pipeline.run(
            [
                "train_models_pipeline",
                "--train_dfs",
                _with_zero_pair_fields(f"{resources_dir}/hg002.train.cnv.h5", f"{tmpdir}/hg002.train.cnv.h5"),
                "--test_dfs",
                _with_zero_pair_fields(f"{resources_dir}/hg002.test.cnv.h5", f"{tmpdir}/hg002.test.cnv.h5"),
                "--output_file_prefix",
                f"{tmpdir}/cnv.model",
                "--gt_type",
                "approximate",
                "--vcf_type",
                "cnv",
                "--verbosity",
                "DEBUG",
            ]
        )
        assert os.path.exists(f"{tmpdir}/cnv.model.h5")
        assert os.path.exists(f"{tmpdir}/cnv.model.pkl")
