import json
from os.path import join as pjoin
from pathlib import Path

import pytest
from ugbio_featuremap import featuremap_xgb_training


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def is_xgb_model(json_file):
    try:
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        # Check for XGBoost-specific keys
        return "learner" in data

    except (json.JSONDecodeError, FileNotFoundError):
        return False  # Not a valid JSON or file doesn't exist


def test_featuremap_xgb_training(tmpdir, resources_dir):
    fp_vcf = pjoin(resources_dir, "fp.chr19.250-500.vcf.gz")
    tp_vcf = pjoin(resources_dir, "tp.chr19.250-500.vcf.gz")
    out_model = pjoin(tmpdir, "model.json")
    featuremap_xgb_training.run(
        [
            "featuremap_xgb_training.py",
            "-tp",
            tp_vcf,
            "-fp",
            fp_vcf,
            "-o",
            out_model,
            "-min_alt_reads",
            "2",
            "-max_alt_reads",
            "3",
            "-chr",
            "chr19",
            "-is_ppm",
            "--split_data_every_2nd_variant",
        ]
    )

    assert is_xgb_model(out_model)
