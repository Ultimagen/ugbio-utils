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
        "--vanilla",
        script_path,
        "-i",
        in_cohort_file,
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

    # Check that full bins have width 2000 (in genomic coordinates, width = end - start + 1)
    # But partial bins at chromosome ends may be smaller
    assert df_rebinned["width"].max() == 2000, "Maximum bin width should be 2000"
    assert (df_rebinned["width"] <= 2000).all(), "No bins should exceed 2000 bp"

    # Check 1-based alignment: all starts should be (N*2000 + 1)
    assert (df_rebinned["start"] % 2000 == 1).all(), "Start positions should align to N*2000+1 (1-based coordinates)"

    # Check that we have approximately half the number of bins
    # (may not be exact due to gaps in coverage and partial bins)
    assert len(df_rebinned) < original_bin_count, "Rebinned cohort should have fewer bins"
    assert len(df_rebinned) <= original_bin_count / 2 + 100, "Should be roughly half the bins"


def test_rebin_cohort_auto_detect(tmpdir, resources_dir, script_path):
    """Test auto-detection of original window length."""
    in_cohort_file = pjoin(resources_dir, "cohort_reads_count.rds")
    out_file = pjoin(tmpdir, "rebinned_cohort_autodetect.rds")
    out_csv = pjoin(tmpdir, "rebinned_cohort_autodetect.csv")

    # No -owl parameter provided
    cmd = [
        "Rscript",
        "--vanilla",
        script_path,
        "-i",
        in_cohort_file,
        "-nwl",
        "2000",
        "-o",
        out_file,
        "--save_csv",
    ]

    result = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True, check=True)

    # Verify auto-detection message in output
    assert "Auto-detected original_window_length" in result.stdout
    assert "1000 bp" in result.stdout or "max width" in result.stdout

    # Verify output correctness
    assert os.path.exists(out_file)
    assert os.path.exists(out_csv)

    df_rebinned = pd.read_csv(out_csv)
    assert df_rebinned["width"].max() == 2000
    assert (df_rebinned["start"] % 2000 == 1).all()


def test_rebin_cohort_equal_window(tmpdir, resources_dir, script_path):
    """Test that equal window sizes just copy without re-binning."""
    in_cohort_file = pjoin(resources_dir, "cohort_reads_count.rds")
    out_file = pjoin(tmpdir, "cohort_same_size.rds")
    out_csv = pjoin(tmpdir, "cohort_same_size.csv")

    cmd = [
        "Rscript",
        "--vanilla",
        script_path,
        "-i",
        in_cohort_file,
        "-nwl",
        "1000",  # Same as original
        "-o",
        out_file,
        "--save_csv",
    ]

    result = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True, check=True)

    # Verify no re-binning message
    assert "No re-binning needed" in result.stdout or "equals original_window_length" in result.stdout

    # Verify files created
    assert os.path.exists(out_file)
    assert os.path.exists(out_csv)

    # Verify data unchanged
    df_output = pd.read_csv(out_csv)
    assert df_output["width"].max() == 1000


def test_rebin_cohort_5x(tmpdir, resources_dir, script_path):
    """Test re-binning from 1000 bp to 5000 bp (5x factor)."""
    in_cohort_file = pjoin(resources_dir, "cohort_reads_count.rds")
    out_file = pjoin(tmpdir, "rebinned_cohort_5000bp.rds")
    out_csv = pjoin(tmpdir, "rebinned_cohort_5000bp.csv")

    cmd = [
        "Rscript",
        "--vanilla",
        script_path,
        "-i",
        in_cohort_file,
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

    # Check that full bins have width 5000 (in genomic coordinates, width = end - start + 1)
    # But partial bins at chromosome ends may be smaller
    assert df_rebinned["width"].max() == 5000, "Maximum bin width should be 5000"
    assert (df_rebinned["width"] <= 5000).all(), "No bins should exceed 5000 bp"

    # CRITICAL TEST for BIOIN-2615: Check 1-based alignment
    # All starts should be (N*5000 + 1), ensuring proper coordinate handling
    assert (df_rebinned["start"] % 5000 == 1).all(), "Start positions should align to N*5000+1 (1-based coordinates)"

    # Verify that some bins are partial (at chromosome ends)
    assert (
        df_rebinned["width"] < 5000
    ).any(), "Should have some partial bins at chromosome ends (not artificially extended)"


def test_rebin_invalid_window_size(tmpdir, resources_dir, script_path):
    """Test that error is raised when new window is not divisible by original."""
    in_cohort_file = pjoin(resources_dir, "cohort_reads_count.rds")
    out_file = pjoin(tmpdir, "rebinned_cohort_invalid.rds")

    cmd = [
        "Rscript",
        "--vanilla",
        script_path,
        "-i",
        in_cohort_file,
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
        "--vanilla",
        script_path,
        "-i",
        in_cohort_file,
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
        "--vanilla",
        script_path,
        "-i",
        in_cohort_file,
        "-nwl",
        "2000",
        "-o",
        out_file,
        "--save_hdf",
    ]

    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    assert os.path.exists(out_file)
    assert os.path.exists(out_hdf5)
