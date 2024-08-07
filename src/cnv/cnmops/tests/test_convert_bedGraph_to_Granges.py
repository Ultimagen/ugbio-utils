import filecmp
import subprocess
from os.path import join as pjoin
from pathlib import Path

from . import get_resource_dir

resources_dir = get_resource_dir(__file__)
base_path = Path(__file__).resolve().parent.parent
script_path = base_path / "convert_bedGraph_to_Granges.R"


def test_convert_bedGraph_to_Granges(tmpdir):
    in_bedGraph_file = pjoin(resources_dir, "test.bedGraph")
    expected_out_file = pjoin(resources_dir, "expected_test.ReadCounts.rds")
    out_file = pjoin(tmpdir, "test.ReadCounts.rds")

    cmd = [
        "Rscript",
        "--vanilla",
        script_path,
        "-i",
        in_bedGraph_file,
        "-sample_name",
        "test"
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    assert filecmp.cmp(out_file, expected_out_file)
