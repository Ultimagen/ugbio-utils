import os
import subprocess
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_FILE = "merge_reads_count_sample_to_cohort.R"


@pytest.fixture
def resources_dir():
    return Path(__file__).parent / "resources"


@pytest.fixture
def script_path():
    base_path = Path(__file__).resolve().parent.parent
    return base_path / SRC_FILE


def test_merge_reads_count_sample_to_cohort(tmpdir, resources_dir, script_path):
    in_cohort_reads_count_file = pjoin(resources_dir, "cohort_gr_obj.rds")
    in_sample_reads_count_file = pjoin(resources_dir, "sample_gr_obj.rds")
    expected_out_merged_reads_count_file = pjoin(resources_dir, "expected_merged_cohort_reads_count.csv")

    out_file = pjoin(tmpdir, "merged_cohort_reads_count.csv")
    os.chdir(tmpdir)
    cmd = [
        "Rscript",
        "--vanilla",
        script_path,
        "-cohort_rc",
        in_cohort_reads_count_file,
        "-sample_rc",
        in_sample_reads_count_file,
        "--save_csv",
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    df = pd.read_csv(out_file)
    df_ref = pd.read_csv(expected_out_merged_reads_count_file)
    assert np.allclose(df.iloc[:, -4], df_ref.iloc[:, -4])
