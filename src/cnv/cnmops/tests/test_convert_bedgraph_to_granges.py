import subprocess
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_FILE = "convert_bedGraph_to_Granges.R"


@pytest.fixture
def resources_dir():
    """Return the path to the test resources directory."""
    return Path(__file__).parent / "resources"


@pytest.fixture
def script_path():
    """Return the path to the R script under test."""
    base_path = Path(__file__).resolve().parent.parent
    return base_path / SRC_FILE


def compare_granges_csv(
    out_csv: str,
    expected_csv: str,
    rtol: float = 1e-9,
) -> bool:
    """
    Compare two GRanges CSV exports with floating-point tolerance.

    Compares string columns exactly and numeric columns using np.allclose()
    to handle minor floating-point differences in serialization.

    Args:
        out_csv: Path to the generated output CSV file.
        expected_csv: Path to the expected CSV file.
        rtol: Relative tolerance for numeric comparison.

    Returns:
        True if the files are semantically equivalent.
    """
    out_df = pd.read_csv(out_csv)
    expected_df = pd.read_csv(expected_csv)

    if out_df.shape != expected_df.shape:
        return False

    # Compare string/object columns exactly
    string_cols = out_df.select_dtypes(include=["object"]).columns
    if not (out_df[string_cols] == expected_df[string_cols]).all().all():
        return False

    # Compare numeric columns with tolerance
    numeric_cols = out_df.select_dtypes(include=[np.number]).columns
    if not np.allclose(out_df[numeric_cols], expected_df[numeric_cols], rtol=rtol):
        return False

    return True


def test_convert_bedgraph_to_granges(tmpdir, resources_dir, script_path):
    """Test conversion of bedGraph file to GRanges RDS format."""
    in_bedgraph_file = pjoin(resources_dir, "test.bedGraph")
    expected_rds_file = pjoin(resources_dir, "expected_test.ReadCounts.rds")
    out_rds_file = pjoin(tmpdir, "test.ReadCounts.rds")
    out_csv_file = pjoin(tmpdir, "test.ReadCounts.csv")
    expected_csv_file = pjoin(tmpdir, "expected_test.ReadCounts.csv")

    # Run the R script to generate the RDS file
    cmd = ["Rscript", script_path, "-i", in_bedgraph_file, "-sample_name", "test"]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0

    # Export both RDS files to CSV for comparison
    # This avoids flaky byte-by-byte comparison of binary RDS files
    # which can differ based on R version or serialization metadata
    export_cmd = [
        "Rscript",
        "-e",
        f"write.csv(as.data.frame(readRDS('{out_rds_file}')), "
        f"'{out_csv_file}', row.names=FALSE); "
        f"write.csv(as.data.frame(readRDS('{expected_rds_file}')), "
        f"'{expected_csv_file}', row.names=FALSE)",
    ]
    assert subprocess.check_call(export_cmd) == 0

    # Compare CSV content with floating-point tolerance
    assert compare_granges_csv(out_csv_file, expected_csv_file)
