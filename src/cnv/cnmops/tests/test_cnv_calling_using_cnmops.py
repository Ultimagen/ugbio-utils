import os
import subprocess
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPT_SRC = "cnv_calling_using_cnmops.R"


@pytest.fixture
def resources_dir():
    return Path(__file__).parent / "resources"


@pytest.fixture
def script_path():
    base_path = Path(__file__).resolve().parent.parent
    return base_path / SCRIPT_SRC


def test_cnv_calling_using_cnmops(tmpdir, resources_dir, script_path):
    in_cohort_reads_count_file = pjoin(resources_dir, "cohort_reads_count.rds")
    expected_out_merged_reads_count_file = pjoin(resources_dir, "expected_cohort.cnmops.cnvs.csv")

    out_file = pjoin(tmpdir, "cohort.cnmops.cnvs.csv")
    os.chdir(tmpdir)
    cmd = [
        "Rscript",
        "--vanilla",
        script_path,
        "-cohort_rc",
        in_cohort_reads_count_file,
        "-minWidth",
        "2",
        "-p",
        "1",
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    compare_files_using_np(expected_out_merged_reads_count_file, out_file)


def compare_files_using_np(expected_out_merged_reads_count_file, out_file):
    out_df = pd.read_csv(out_file)
    df_ref = pd.read_csv(expected_out_merged_reads_count_file)
    # Separate numeric and string columns
    numeric_cols = out_df.select_dtypes(include=[np.number]).columns
    string_cols = out_df.select_dtypes(include=[object]).columns
    # Compare numeric columns using np.allclose
    numeric_comparison = np.allclose(out_df[numeric_cols], df_ref[numeric_cols])
    # Compare string columns using regular equality
    string_comparison = (out_df[string_cols] == df_ref[string_cols]).all().all()
    # Combine the results
    assert numeric_comparison and string_comparison
