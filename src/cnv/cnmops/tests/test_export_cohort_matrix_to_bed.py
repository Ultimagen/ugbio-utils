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

    cmd = ["Rscript", "--vanilla", script_path, input_rds]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0

    # Check that BED files were created for each sample
    bed_files = list(Path(tmpdir).glob("*.cov.bed"))
    assert len(bed_files) > 0, "No BED files were created"

    # Verify BED file format (should have chrom, start, end, score columns)
    for bed_file in bed_files:
        df_test = pd.read_csv(bed_file, sep="\t", header=None)
        assert len(df_test.columns) >= 4, f"BED file {bed_file} should have at least 4 columns"
        assert len(df_test) > 0, f"BED file {bed_file} should not be empty"


def test_export_cohort_matrix_to_bed_mean(tmpdir, resources_dir, script_path):
    """Test exporting mean cohort coverage BED file."""
    input_rds = pjoin(resources_dir, "merged_cohort_reads_count.rds")

    cmd = ["Rscript", "--vanilla", script_path, input_rds, "--mean"]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0

    # Check that cohort BED file was created
    cohort_bed = Path(tmpdir) / "coverage.cohort.bed"
    assert cohort_bed.exists(), "Cohort BED file was not created"

    # Verify BED file format
    df_test = pd.read_csv(cohort_bed, sep="\t", header=None)
    assert len(df_test.columns) >= 4, "BED file should have at least 4 columns"
    assert len(df_test) > 0, "BED file should not be empty"
