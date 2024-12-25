import subprocess
from os.path import join as pjoin
from pathlib import Path

import pytest

BASE_PATH = Path(__file__).parent.parent.parent
REPORT_NOTEBOOK = BASE_PATH / "ugbio_methylation" / "reports" / "methyldackel_qc_report.ipynb"


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def test_methyldackel_create_qc_report(tmpdir, resources_dir):
    papermill_out = pjoin(tmpdir, "methyldackel_qc_report.papermill.ipynb")
    input_h5_file = pjoin(resources_dir, "input_for_html_report.h5")
    base_file_name = "test"

    cmd1 = (
        f"papermill {REPORT_NOTEBOOK} {papermill_out} "
        f"-p input_h5_file {input_h5_file} "
        f"-p input_base_file_name {base_file_name}"
    )
    assert subprocess.check_call(cmd1.split(), cwd=tmpdir) == 0

    cmd2 = f"jupyter nbconvert --to html {papermill_out} --template classic --no-input --output {base_file_name}.html"
    assert subprocess.check_call(cmd2.split(), cwd=tmpdir) == 0
