from pathlib import Path

import pytest
from ugbio_core import picard_metrics_utils


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def test_parse_cvg_metrics(tmpdir, resources_dir):
    picard_file = resources_dir / "140479-BC21.alignment_summary_metrics"
    metric_df = picard_metrics_utils.parse_cvg_metrics(picard_file)[1]
    histogram_df = picard_metrics_utils.parse_cvg_metrics(picard_file)[2]
    assert picard_metrics_utils.parse_cvg_metrics(picard_file)[0] == "AlignmentSummaryMetrics"
    assert metric_df.TOTAL_READS[0] == 640936116
    assert histogram_df.READ_LENGTH[4] == 29
