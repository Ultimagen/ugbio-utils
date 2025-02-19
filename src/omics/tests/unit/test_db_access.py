import pickle
from pathlib import Path

import pandas as pd
import pytest
from ugbio_omics.db_access import inputs2df, metrics2df, nexus_metrics_to_df


HARDCODED_WFIDS = [
    "de06922f-07f8-4b51-843e-972308c81c6f",
    "ea5e54d8-3db1-47b2-bc0e-68f38e3e89f3",
    "9588412a-e8c2-447d-b1e7-dc46b5da3fb4",
    "469b7436-a737-4257-9f08-7990ff95a461",
    "0440f417-be57-4887-b668-39c47cbd55aa",
    "2989cdc2-6fa5-4931-adcf-a9c7372f162a",
    "ddc419d2-5b94-4b92-85e0-6f17c56f3e4d",
    "6674c3cc-410d-47ec-9718-b23fa34c86e1",
    "6289c2fa-0c6a-4de8-805e-cae0e227227f",
    "1e4440f9-76a3-4bd0-8359-8c653d5f7212",
]


@pytest.fixture
def resources_dir():
    inputs_dir = Path(__file__).parent.parent / "resources"
    return inputs_dir


def test_metrics2df(resources_dir):
    docs = pickle.load(open(resources_dir / "test_fetch_from_database_query1.pkl", "rb"))
    docs = sorted(docs, key=lambda x: x["metadata"]["workflowId"])
    docs = [x for x in docs if x["metadata"]["workflowId"] in HARDCODED_WFIDS]
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

    all_metrics = pd.concat((metrics2df(x, metrics_to_report) for x in docs), axis=0)
    expected_df = pd.read_hdf(resources_dir / "expected_metrics_df.h5", key="df")
    print(all_metrics.compare(expected_df, keep_equal=False))
    pd.testing.assert_frame_equal(all_metrics, expected_df, check_dtype=False)


def test_inputs_outputs_dataframe(resources_dir):
    docs = pickle.load(open(resources_dir / "test_fetch_from_database_query1.pkl", "rb"))
    docs = sorted(docs, key=lambda x: x["metadata"]["workflowId"])
    docs = [x for x in docs if x["metadata"]["workflowId"] in HARDCODED_WFIDS]
    all_inputs = pd.concat((inputs2df(x) for x in docs), axis=0)
    expected_df = pd.read_hdf(resources_dir / "expected_inputs_df.h5", key="df")
    pd.testing.assert_frame_equal(all_inputs, expected_df, check_dtype=False)


def test_nexus_metrics_to_df(resources_dir):
    docs = pickle.load(open(resources_dir / "test_fetch_from_database_query_nexus_inputs.pkl", "rb"))
    assert len(docs) == 23
    nexus_metrics = pd.concat(nexus_metrics_to_df(x) for x in docs)
    expected_df = pd.read_hdf(resources_dir / "expected_nexus_df.h5", key="df")
    pd.testing.assert_frame_equal(nexus_metrics, expected_df, check_dtype=False)


def test_omics_inputs(resources_dir):
    docs = pickle.load(open(resources_dir / "test_fetch_from_database_query_omics_inputs.pkl", "rb"))
    omics_inputs = pd.concat(inputs2df(x) for x in docs)
    omics_inputs = omics_inputs.sort_index()
    hdf = pd.read_hdf(resources_dir / "expected_omics_df.h5").astype(omics_inputs.dtypes)
    pd.testing.assert_frame_equal(omics_inputs, hdf, check_dtype=False)

