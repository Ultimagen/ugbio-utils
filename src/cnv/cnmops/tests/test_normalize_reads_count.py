import os
import subprocess
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_FILE = "normalize_reads_count.R"


@pytest.fixture
def resources_dir():
    return Path(__file__).parent / "resources"


@pytest.fixture
def script_path():
    base_path = Path(__file__).resolve().parent.parent
    return base_path / SRC_FILE


def test_normalize_reads_count(tmpdir, resources_dir, script_path):
    in_cohort_reads_count_file = pjoin(resources_dir, "test_rc.rds")
    expected_out_norm_rc = pjoin(resources_dir, "test_rc.norm.cohort_reads_count.norm.csv")

    out_file = pjoin(tmpdir, "cohort_reads_count.norm.csv")
    os.chdir(tmpdir)
    cmd = ["Rscript", "--vanilla", script_path, "-cohort_reads_count_file", in_cohort_reads_count_file, "--save_csv"]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    df = pd.read_csv(out_file)
    df_ref = pd.read_csv(expected_out_norm_rc)
    assert np.allclose(df.iloc[:, -6], df_ref.iloc[:, -6])


def test_normalize_reads_count_with_ploidy(tmpdir, resources_dir, script_path):
    in_cohort_reads_count_file = pjoin(resources_dir, "test_rc.rds")
    ploidy_file = pjoin(resources_dir, "test_rc.ploidy")
    expected_out_norm_rc = pjoin(resources_dir, "test_rc.norm.cohort_reads_count.norm.csv")

    out_file = pjoin(tmpdir, "cohort_reads_count.norm.csv")
    os.chdir(tmpdir)
    cmd = [
        "Rscript",
        "--vanilla",
        script_path,
        "-cohort_reads_count_file",
        in_cohort_reads_count_file,
        "-ploidy",
        ploidy_file,
        "--save_csv",
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    df = pd.read_csv(out_file)
    df_ref = pd.read_csv(expected_out_norm_rc)
    assert np.allclose(df.iloc[:, -6], df_ref.iloc[:, -6])


def test_normalize_reads_count_without_chrX(tmpdir, resources_dir, script_path):
    in_cohort_reads_count_file = pjoin(resources_dir, "test_rc.noX.rds")
    expected_out_norm_rc = pjoin(resources_dir, "cohort_reads_count_noX.norm.csv")

    out_file = pjoin(tmpdir, "cohort_reads_count.norm.csv")
    os.chdir(tmpdir)
    cmd = ["Rscript", "--vanilla", script_path, "-cohort_reads_count_file", in_cohort_reads_count_file, "--save_csv"]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    df = pd.read_csv(out_file)
    df_ref = pd.read_csv(expected_out_norm_rc)
    assert np.allclose(df.iloc[:, -6], df_ref.iloc[:, -6])


def test_normalize_reads_count_without_chrXchrY(tmpdir, resources_dir, script_path):
    in_cohort_reads_count_file = pjoin(resources_dir, "test_rc.noXnoY.rds")
    expected_out_norm_rc = pjoin(resources_dir, "cohort_reads_count.norm.noXnoY.csv")

    out_file = pjoin(tmpdir, "cohort_reads_count.norm.csv")
    os.chdir(tmpdir)
    cmd = ["Rscript", "--vanilla", script_path, "-cohort_reads_count_file", in_cohort_reads_count_file, "--save_csv"]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    df = pd.read_csv(out_file)
    df_ref = pd.read_csv(expected_out_norm_rc)
    assert np.allclose(df.iloc[:, -6], df_ref.iloc[:, -6])
