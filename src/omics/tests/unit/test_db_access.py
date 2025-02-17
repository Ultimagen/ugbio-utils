from ugbio_omics.db_access import inputs2df, metrics2df, nexus_metrics_to_df


def test_metrics2df():
    doc = {
        "metadata": {
            "workflowId": "test123",
            "workflowEntity": "sample",
            "entityType": "someEntity",  # Added entityType
        },
        "metrics": {"AlignmentSummaryMetrics": [{"metric1": 0.5, "metric2": 0.8}]},
    }
    metrics_df = metrics2df(doc, ["AlignmentSummaryMetrics"])
    assert "AlignmentSummaryMetrics" in metrics_df.columns.levels[0]


def test_inputs2df():
    doc = {
        "metadata": {"workflowId": "test123", "workflowEntity": "sample", "entityType": "someEntity"},
        "inputs": {"input1": "value1"},
        "outputs": {"output1": "value2"},
    }
    input_df = inputs2df(doc)
    assert "input1" in input_df.columns
    assert "output1" in input_df.columns


def test_nexus_metrics_to_df():
    input_dict = {
        "_id": "some_id",
        "metadata_sequencingRunId": "run123",
        "metric_x_1": 10,
        "metric_y_2": 20,
    }
    metrics_df = nexus_metrics_to_df(input_dict)
    assert metrics_df.index.name == "run123"
