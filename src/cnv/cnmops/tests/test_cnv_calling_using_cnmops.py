import os
import subprocess
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd

from . import get_resource_dir

resources_dir = get_resource_dir(__file__)
base_path = Path(__file__).resolve().parent.parent
script_path = base_path / "cnv_calling_using_cnmops.R"


def test_cnv_calling_using_cnmops(tmpdir):
    in_cohort_reads_count_file = pjoin(resources_dir, "merged_cohort_reads_count.rds")
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
    df = pd.read_csv(out_file)
    df_ref = pd.read_csv(expected_out_merged_reads_count_file)
    # Separate numeric and string columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    string_cols = df.select_dtypes(include=[object]).columns
    # Compare numeric columns using np.allclose
    numeric_comparison = np.allclose(df[numeric_cols], df_ref[numeric_cols])
    # Compare string columns using regular equality
    string_comparison = (df[string_cols] == df_ref[string_cols]).all().all()
    # Combine the results
    assert numeric_comparison and string_comparison
