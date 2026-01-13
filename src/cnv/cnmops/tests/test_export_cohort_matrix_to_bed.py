import subprocess
from os.path import join as pjoin
from pathlib import Path

import pandas as pd
import pytest

SRC_FILE = "export_cohort_matrix_to_bed.R"


@pytest.fixture
def resources_dir():
    return Path(__file__).parent / "resources"


@pytest.fixture
def script_path():
    base_path = Path(__file__).resolve().parent.parent
    return base_path / SRC_FILE


def test_export_cohort_matrix_to_bed_per_sample(tmpdir, resources_dir, script_path):
    """Test exporting per-sample coverage BED files from cohort matrix."""
    input_rds = pjoin(resources_dir, "merged_cohort_reads_count.rds")
    expected_bed = pjoin(resources_dir, "expected_test.bam.cov.bed")

    cmd = ["Rscript", "--vanilla", script_path, input_rds]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0

    # Check that BED files were created for each sample
    bed_files = list(Path(tmpdir).glob("*.cov.bed"))
    assert len(bed_files) > 0, "No BED files were created"

    # Compare against expected output
    actual_bed = Path(tmpdir) / "test.bam.cov.bed"
    assert actual_bed.exists(), "Expected BED file test.bam.cov.bed was not created"

    df_actual = pd.read_csv(actual_bed, sep="\t", header=None)
    df_expected = pd.read_csv(expected_bed, sep="\t", header=None)

    # Compare dataframes
    pd.testing.assert_frame_equal(df_actual, df_expected, check_dtype=False)


def test_export_cohort_matrix_to_bed_mean(tmpdir, resources_dir, script_path):
    """Test exporting mean cohort coverage BED file."""
    input_rds = pjoin(resources_dir, "merged_cohort_reads_count.rds")
    expected_bed = pjoin(resources_dir, "expected_coverage.cohort.bed")

    cmd = ["Rscript", "--vanilla", script_path, input_rds, "--mean"]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0

    # Check that cohort BED file was created
    actual_bed = Path(tmpdir) / "coverage.cohort.bed"
    assert actual_bed.exists(), "Cohort BED file was not created"

    # Compare against expected output
    df_actual = pd.read_csv(actual_bed, sep="\t", header=None)
    df_expected = pd.read_csv(expected_bed, sep="\t", header=None)

    # Compare dataframes
    pd.testing.assert_frame_equal(df_actual, df_expected, check_dtype=False)
