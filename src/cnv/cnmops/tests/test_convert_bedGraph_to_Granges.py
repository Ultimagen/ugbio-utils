import filecmp
import subprocess
from os.path import join as pjoin
from pathlib import Path

import pytest

SRC_FILE = "convert_bedGraph_to_Granges.R"


@pytest.fixture
def resources_dir():
    return Path(__file__).parent / "resources"


@pytest.fixture
def script_path():
    base_path = Path(__file__).resolve().parent.parent
    return base_path / SRC_FILE


def test_convert_bedGraph_to_Granges(tmpdir, resources_dir, script_path):
    in_bedGraph_file = pjoin(resources_dir, "test.bedGraph")
    expected_out_file = pjoin(resources_dir, "expected_test.ReadCounts.rds")
    out_file = pjoin(tmpdir, "test.ReadCounts.rds")

    cmd = ["Rscript", "--vanilla", script_path, "-i", in_bedGraph_file, "-sample_name", "test"]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    assert filecmp.cmp(out_file, expected_out_file)
