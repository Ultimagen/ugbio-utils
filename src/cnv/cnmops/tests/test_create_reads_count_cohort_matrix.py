import subprocess
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_FILE = "create_reads_count_cohort_matrix.R"


@pytest.fixture
def resources_dir():
    return Path(__file__).parent / "resources"


@pytest.fixture
def script_path():
    base_path = Path(__file__).resolve().parent.parent
    return base_path / SRC_FILE


def test_create_reads_count_cohort_matrix(tmpdir, resources_dir, script_path):
    in_rds_file = pjoin(resources_dir, "sample_gr_obj.rds")
    rc_files_list = pjoin(tmpdir, "samples_read_count_files_list")
    with open(rc_files_list, "w") as f:
        f.write(in_rds_file + "\n")
        f.write(in_rds_file + "\n")
        f.write(in_rds_file + "\n")

    expected_out_file = pjoin(resources_dir, "merged_cohort_reads_count.csv")
    out_file = pjoin(tmpdir, "merged_cohort_reads_count.csv")

    cmd = ["Rscript", "--vanilla", script_path, "-samples_read_count_files_list", rc_files_list, "--save_csv"]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    result_df = pd.read_csv(out_file)
    df_ref = pd.read_csv(expected_out_file)
    assert np.allclose(result_df.iloc[:, -3:], df_ref.iloc[:, -3:])
