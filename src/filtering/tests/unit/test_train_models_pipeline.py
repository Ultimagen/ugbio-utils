import os
from pathlib import Path

import pytest
from ugbio_filtering import train_models_pipeline


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
