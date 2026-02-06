"""Tests for rebin_cohort_reads_count.R script.

Note: These tests require R with cn.mops package installed.
Tests will be skipped if R environment is not properly configured.
"""

import os
import subprocess
from os.path import join as pjoin
from pathlib import Path

import pandas as pd
import pytest

SRC_FILE = "rebin_cohort_reads_count.R"


def check_r_environment():
    """Check if R and cn.mops are available."""
    try:
        result = subprocess.run(
            ["Rscript", "-e", "suppressPackageStartupMessages(library(cn.mops))"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


# Skip all tests if R environment is not available
pytestmark = pytest.mark.skipif(not check_r_environment(), reason="R with cn.mops package not available")


@pytest.fixture
def resources_dir():
    return Path(__file__).parent / "resources"


@pytest.fixture
def script_path():
    base_path = Path(__file__).resolve().parent.parent
    return base_path / SRC_FILE


def test_rebin_cohort_2x(tmpdir, resources_dir, script_path):
    """Test re-binning from 1000 bp to 2000 bp (2x factor)."""
    in_cohort_file = pjoin(resources_dir, "cohort_reads_count.rds")
    out_file = pjoin(tmpdir, "rebinned_cohort_2000bp.rds")
    out_csv = pjoin(tmpdir, "rebinned_cohort_2000bp.csv")

    cmd = [
        "Rscript",
        script_path,
        "-i",
        in_cohort_file,
        "-owl",
        "1000",
        "-nwl",
        "2000",
        "-o",
        out_file,
        "--save_csv",
    ]

    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    assert os.path.exists(out_file)
    assert os.path.exists(out_csv)

    # Load rebinned data
    df_rebinned = pd.read_csv(out_csv)

    # The original cohort has 1992 bins (as shown in script output)
    original_bin_count = 1992

    # Check that bin widths are 2001 (in R IRanges, width = end - start + 1)
    assert (df_rebinned["width"] == 2001).all(), "All bins should have width 2001"

    # Check that we have approximately half the number of bins
    # (may not be exact due to gaps in coverage)
    assert len(df_rebinned) < original_bin_count, "Rebinned cohort should have fewer bins"
    assert len(df_rebinned) <= original_bin_count / 2 + 100, "Should be roughly half the bins"


def test_rebin_cohort_5x(tmpdir, resources_dir, script_path):
    """Test re-binning from 1000 bp to 5000 bp (5x factor)."""
    in_cohort_file = pjoin(resources_dir, "cohort_reads_count.rds")
    out_file = pjoin(tmpdir, "rebinned_cohort_5000bp.rds")
    out_csv = pjoin(tmpdir, "rebinned_cohort_5000bp.csv")

    cmd = [
        "Rscript",
        script_path,
        "-i",
        in_cohort_file,
        "-owl",
        "1000",
        "-nwl",
        "5000",
        "-o",
        out_file,
        "--save_csv",
    ]

    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    assert os.path.exists(out_file)
    assert os.path.exists(out_csv)

    # Load rebinned data
    df_rebinned = pd.read_csv(out_csv)

    # Check that bin widths are 5001 (in R IRanges, width = end - start + 1)
    assert (df_rebinned["width"] == 5001).all(), "All bins should have width 5001"

    # Check that bins are aligned to 5000 bp boundaries
    assert (df_rebinned["start"] % 5000 == 0).all(), "Start positions should be multiples of 5000"


def test_rebin_invalid_window_size(tmpdir, resources_dir, script_path):
    """Test that error is raised when new window is not divisible by original."""
    in_cohort_file = pjoin(resources_dir, "cohort_reads_count.rds")
    out_file = pjoin(tmpdir, "rebinned_cohort_invalid.rds")

    cmd = [
        "Rscript",
        script_path,
        "-i",
        in_cohort_file,
        "-owl",
        "1000",
        "-nwl",
        "1500",  # Not divisible by 1000
        "-o",
        out_file,
    ]

    # Should fail with non-zero exit code
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_call(cmd, cwd=tmpdir)


def test_rebin_smaller_window_fails(tmpdir, resources_dir, script_path):
    """Test that error is raised when new window is smaller than original."""
    in_cohort_file = pjoin(resources_dir, "cohort_reads_count.rds")
    out_file = pjoin(tmpdir, "rebinned_cohort_smaller.rds")

    cmd = [
        "Rscript",
        script_path,
        "-i",
        in_cohort_file,
        "-owl",
        "1000",
        "-nwl",
        "500",  # Smaller than original
        "-o",
        out_file,
    ]

    # Should fail with non-zero exit code
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.check_call(cmd, cwd=tmpdir)


def test_rebin_hdf5_output(tmpdir, resources_dir, script_path):
    """Test HDF5 output format."""
    in_cohort_file = pjoin(resources_dir, "cohort_reads_count.rds")
    out_file = pjoin(tmpdir, "rebinned_cohort_hdf5.rds")
    out_hdf5 = pjoin(tmpdir, "rebinned_cohort_hdf5.hdf5")

    cmd = [
        "Rscript",
        script_path,
        "-i",
        in_cohort_file,
        "-owl",
        "1000",
        "-nwl",
        "2000",
        "-o",
        out_file,
        "--save_hdf",
    ]

    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    assert os.path.exists(out_file)
    assert os.path.exists(out_hdf5)
