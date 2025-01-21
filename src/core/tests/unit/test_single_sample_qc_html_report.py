from pathlib import Path

import pytest
from ugbio_core.reports.report_utils import generate_report

BASE_PATH = Path(__file__).parent.parent.parent
REPORT_BASE_PATH = BASE_PATH / "ugbio_core" / "reports"
REPORT_NOTEBOOK = REPORT_BASE_PATH / "single_sample_qc_create_html_report.ipynb"


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def test_single_sample_qc_create_html_report(tmpdir, resources_dir):
    input_h5_file = resources_dir / "input_for_html_report.h5"
    base_file_name = "test"

    parameters = {
        "top_metrics_file": f"{REPORT_BASE_PATH}/top_metrics_for_tbl.csv",
        "input_h5_file": input_h5_file,
        "input_base_file_name": base_file_name,
    }

    report_html = Path(tmpdir) / f"{base_file_name}.html"

    generate_report(template_notebook_path=REPORT_NOTEBOOK, parameters=parameters, output_report_html_path=report_html)

    assert report_html.exists()
    assert report_html.stat().st_size > 0
