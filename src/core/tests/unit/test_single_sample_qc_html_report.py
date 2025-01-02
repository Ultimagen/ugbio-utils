import subprocess
from os.path import join as pjoin
from pathlib import Path

import pytest

BASE_PATH = Path(__file__).parent.parent.parent
REPORT_BASE_PATH = BASE_PATH / "ugbio_core" / "reports"
REPORT_NOTEBOOK = REPORT_BASE_PATH / "single_sample_qc_create_html_report.ipynb"


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def test_single_sample_qc_create_html_report(tmpdir, resources_dir):
    papermill_out = pjoin(tmpdir, "single_sample_qc_create_html_report.papermill.ipynb")
    input_h5_file = pjoin(resources_dir, "input_for_html_report.h5")
    base_file_name = "test"

    cmd = (
        f"papermill {REPORT_NOTEBOOK} {papermill_out} "
        f"-p top_metrics_file {REPORT_BASE_PATH}/top_metrics_for_tbl.csv "
        f"-p input_h5_file {input_h5_file} "
        f"-p input_base_file_name {base_file_name}"
    )

    assert subprocess.check_call(cmd.split(), cwd=tmpdir) == 0

    jupyter_convert_cmd = (
        f"jupyter nbconvert --to html {papermill_out} --template classic --no-input --output {base_file_name}.html"
    )
    assert subprocess.check_call(jupyter_convert_cmd.split(), cwd=tmpdir) == 0
