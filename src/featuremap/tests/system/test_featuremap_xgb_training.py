import json
from os.path import join as pjoin
from pathlib import Path

import pytest
import ugbio_featuremap.featuremap_xgb_training as featuremap_xgb_training


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def load_json(file_path):
    with open(file_path) as f:
        return json.load(f)


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
        ]
    )

    expected_model_file = pjoin(resources_dir, "expected_model_alt_reads_2_3.v2.json")

    expected_json = load_json(expected_model_file)
    output_json = load_json(out_model)

    assert expected_json == output_json
