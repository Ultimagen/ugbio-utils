import subprocess
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_FILE = "get_reads_count_from_bam.R"


@pytest.fixture
def resources_dir():
    return Path(__file__).parent / "resources"


@pytest.fixture
def script_path():
    base_path = Path(__file__).resolve().parent.parent
    return base_path / SRC_FILE


def test_get_reads_count_from_bam(tmpdir, resources_dir, script_path):
    in_bam_file = pjoin(resources_dir, "test.bam")
    expected_out_file = pjoin(resources_dir, "test.ReadCounts.csv")
    out_prefix = pjoin(tmpdir, "out_test")
    out_file = pjoin(tmpdir, "out_test.ReadCounts.csv")

    cmd = [
        "Rscript",
        "--vanilla",
        script_path,
        "-i",
        in_bam_file,
        "-refseq",
        "chr1",
        "-wl",
        "1000",
        "-p",
        "1",
        "-o",
        out_prefix,
        "--save_csv",
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    result_df = pd.read_csv(out_file)
    expected_df = pd.read_csv(expected_out_file)
    assert np.allclose(result_df.iloc[:, -1], expected_df.iloc[:, -1])
