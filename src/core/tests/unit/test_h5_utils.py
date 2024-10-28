from pathlib import Path

import numpy as np
import pytest
from ugbio_core import h5_utils


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def test_preprocess_h5_key_with_slash():
    assert h5_utils.preprocess_h5_key("/foo") == "foo"


def test_preprocess_h5_key_without_slash():
    assert h5_utils.preprocess_h5_key("foo") == "foo"


def test_should_skip_h5_key_true():
    assert h5_utils.should_skip_h5_key("str_histogram_123str", "histogram")


def test_should_skip_h5_key_false():
    assert not h5_utils.should_skip_h5_key("str_his_togram_123str", "histogram")


def test_get_h5_keys(resources_dir):
    metrics_h5_path = str(resources_dir / "140479-BC21_aggregated_metrics.h5")
    assert np.array_equal(
        h5_utils.get_h5_keys(metrics_h5_path),
        [
            "/AlignmentSummaryMetrics",
            "/DuplicationMetrics",
            "/GcBiasDetailMetrics",
            "/GcBiasSummaryMetrics",
            "/QualityYieldMetrics",
            "/RawWgsMetrics",
            "/WgsMetrics",
            "/histogram_AlignmentSummaryMetrics",
            "/histogram_RawWgsMetrics",
            "/histogram_WgsMetrics",
            "/histogram_coverage",
            "/stats_coverage",
        ],
    )


def test_convert_h5_to_json(resources_dir):
    metrics_h5_path = str(resources_dir / "140479-BC21_aggregated_metrics.h5")
    metrics_json_path = str(resources_dir / "140479-BC21_aggregated_metrics.json")
    with open(metrics_json_path) as json_file:
        data = json_file.read()
    assert f'{h5_utils.convert_h5_to_json(metrics_h5_path, "metrics", "histogram")}\n' == data
