from unittest.mock import MagicMock, patch

import pandas as pd
from ugbio_omics.db_access import inputs2df, metrics2df, nexus_metrics_to_df, query_database


@patch("ugbio_omics.db_access.collections")
def test_query_database(self, mock_collections):
    mock_collection = MagicMock()
    mock_collections.__getitem__.return_value = mock_collection
    mock_collection.find.return_value = [{"_id": 1, "name": "test"}]

    result = query_database({}, "pipelines")
    self.assertEqual(result, [{"_id": 1, "name": "test"}])
    mock_collection.find.assert_called_once_with({}, {})


def test_metrics2df(self):
    doc = {
        "metadata": {"workflowId": "test123", "workflowEntity": "sample"},
        "metrics": {"AlignmentSummaryMetrics": [{"metric1": 0.5, "metric2": 0.8}]},
    }
    metrics_df = metrics2df(doc, ["AlignmentSummaryMetrics"])
    self.assertIsInstance(metrics_df, pd.DataFrame)
    self.assertIn("AlignmentSummaryMetrics", metrics_df.columns.levels[0])


def test_inputs2df(self):
    doc = {
        "metadata": {"workflowId": "test123", "workflowEntity": "sample"},
        "inputs": {"input1": "value1"},
        "outputs": {"output1": "value2"},
    }
    input_df = inputs2df(doc)
    self.assertIsInstance(input_df, pd.DataFrame)
    self.assertIn("input1", input_df.columns)
    self.assertIn("output1", input_df.columns)


def test_nexus_metrics_to_df(self):
    input_dict = {
        "_id": "some_id",
        "metadata_sequencingRunId": "run123",
        "metric_x_1": 10,
        "metric_y_2": 20,
    }
    metrics_df = nexus_metrics_to_df(input_dict)
    self.assertIsInstance(metrics_df, pd.DataFrame)
    self.assertEqual(metrics_df.index.name, "run123")
