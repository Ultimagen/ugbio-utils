import json
from pathlib import Path

import pandas as pd
import pytest
from ugbio_omics.db_access import metrics2df


@pytest.fixture
def resources_dir():
    inputs_dir = Path(__file__).parent.parent / "resources"
    return inputs_dir


def test_metrics2df(resources_dir):
    with open(resources_dir / "db_doc.json") as f:
        doc = json.load(f)
    metrics_to_report = [
        "AlignmentSummaryMetrics",
        "Contamination",
        "DuplicationMetrics",
        "GcBiasDetailMetrics",
        "GcBiasSummaryMetrics",
        "QualityYieldMetrics",
        "RawWgsMetrics",
        "WgsMetrics",
        "stats_coverage",
        "short_report_/all_data",
        "short_report_/all_data_gt",
    ]

    metrics_df = metrics2df(doc, metrics_to_report)
    assert metrics_df.equals(pd.read_hdf(resources_dir / "expected_metrics_df.h5", key="df"))


# def test_inputs2df(resources_dir):
#     with open(resources_dir / "db_doc.json") as f:
#         doc = json.load(f)
#     inputs_outputs_df = (inputs2df(doc))
#     assert inputs_outputs_df[""]
#
#
# def test_nexus_metrics_to_df():
#     input_dict = {
#         "_id": "some_id",
#         "metadata_sequencingRunId": "run123",
#         "metric_x_1": 10,
#         "metric_y_2": 20,
#     }
#     metrics_df = nexus_metrics_to_df(input_dict)
#     assert metrics_df.index.name == "run123"
