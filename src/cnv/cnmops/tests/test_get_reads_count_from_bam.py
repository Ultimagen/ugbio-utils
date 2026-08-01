import subprocess
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SRC_FILE = "get_reads_count_from_bam.R"
EXPORT_SRC_FILE = "export_cohort_matrix_to_bed.R"


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


# Skip all tests if R environment is not available (tests run inside the ugbio_cnv docker image)
pytestmark = pytest.mark.skipif(not check_r_environment(), reason="R with cn.mops package not available")


@pytest.fixture
def resources_dir():
    return Path(__file__).parent / "resources"


@pytest.fixture
def script_path():
    base_path = Path(__file__).resolve().parent.parent
    return base_path / SRC_FILE


@pytest.fixture
def export_script_path():
    base_path = Path(__file__).resolve().parent.parent
    return base_path / EXPORT_SRC_FILE


def test_get_reads_count_from_bam(tmpdir, resources_dir, script_path):
    in_bam_file = pjoin(resources_dir, "test.bam")
    expected_out_file = pjoin(resources_dir, "test.ReadCounts.csv")
    out_prefix = pjoin(tmpdir, "out_test")
    out_file = pjoin(tmpdir, "out_test.ReadCounts.csv")

    cmd = [
        "Rscript",
        "--vanilla",
        script_path,
        "-i",
        in_bam_file,
        "-refseq",
        "chr1",
        "-wl",
        "1000",
        "-p",
        "1",
        "-o",
        out_prefix,
        "--save_csv",
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0
    result_df = pd.read_csv(out_file)
    expected_df = pd.read_csv(expected_out_file)
    assert np.allclose(result_df.iloc[:, -1], expected_df.iloc[:, -1])


def test_get_reads_count_from_bam_intervals(tmpdir, resources_dir, script_path, export_script_path):
    """--intervals mode must agree with -refseq/-wl on identical windows.

    Derive intervals.bed from the committed chr1 cohort fixture (merged_cohort_reads_count.rds,
    chr1 @ 1000bp windows) via export_cohort_matrix_to_bed.R --intervals_only, count reads over
    those exact windows, and compare against the full-chr1 expectation (test.ReadCounts.csv).
    """
    in_bam_file = pjoin(resources_dir, "test.bam")
    cohort_rds = pjoin(resources_dir, "merged_cohort_reads_count.rds")
    expected_out_file = pjoin(resources_dir, "test.ReadCounts.csv")
    intervals_bed = pjoin(tmpdir, "intervals.bed")
    out_prefix = pjoin(tmpdir, "out_intervals")
    out_file = pjoin(tmpdir, "out_intervals.ReadCounts.csv")

    # 1. Build the cohort windows BED (BED3, 0-based start) from the cohort RDS.
    export_cmd = ["Rscript", "--vanilla", export_script_path, cohort_rds, "--intervals_only"]
    assert subprocess.check_call(export_cmd, cwd=tmpdir) == 0
    assert Path(intervals_bed).exists(), "intervals.bed was not created"

    # 2. Count reads over exactly those windows.
    cmd = [
        "Rscript",
        "--vanilla",
        script_path,
        "-i",
        in_bam_file,
        "--intervals",
        intervals_bed,
        "-p",
        "1",
        "-o",
        out_prefix,
        "--save_csv",
    ]
    assert subprocess.check_call(cmd, cwd=tmpdir) == 0

    # 3. Counts must match the -refseq/-wl expectation on the same (seqnames, start, end) windows.
    #    The count column is the last one in each CSV (named after the BAM basename, "test.bam").
    result_df = pd.read_csv(out_file)
    expected_df = pd.read_csv(expected_out_file)
    keys = ["seqnames", "start", "end"]
    result_counts = result_df[[*keys, result_df.columns[-1]]].rename(columns={result_df.columns[-1]: "count_actual"})
    expected_counts = expected_df[[*keys, expected_df.columns[-1]]].rename(
        columns={expected_df.columns[-1]: "count_expected"}
    )
    merged = result_counts.merge(expected_counts, on=keys)
    assert len(merged) == len(result_df), "intervals windows are not a subset of the expected windows"
    assert np.allclose(merged["count_actual"], merged["count_expected"])


def test_get_reads_count_from_bam_intervals_mutually_exclusive(tmpdir, resources_dir, script_path):
    """--intervals combined with an explicit -refseq/-wl must error out."""
    in_bam_file = pjoin(resources_dir, "test.bam")
    intervals_bed = pjoin(resources_dir, "test.bam")  # any path; error triggers before it is read
    out_prefix = pjoin(tmpdir, "out_bad")

    cmd = [
        "Rscript",
        "--vanilla",
        script_path,
        "-i",
        in_bam_file,
        "--intervals",
        intervals_bed,
        "-wl",
        "1000",
        "-o",
        out_prefix,
    ]
    result = subprocess.run(cmd, cwd=tmpdir, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "mutually exclusive" in result.stderr
