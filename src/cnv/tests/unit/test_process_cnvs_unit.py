import filecmp
from os.path import join as pjoin
from pathlib import Path

import pytest
from ugbio_cnv.cnv_bed_format_utils import aggregate_annotations_in_df, annotate_bed


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


class TestProcessCnvs:
    def test_annotate_bed(self, tmpdir, resources_dir):
        input_bed_file = pjoin(resources_dir, "unfiltered_cnvs.bed")
        expected_out_annotate_bed_file = pjoin(resources_dir, "annotate_cnv.bed")
        coverage_lcr_file = pjoin(resources_dir, "UG-CNV-LCR.bed")
        intersection_cutoff = 0.5
        min_cnv_length = 10000
        prefix = f"{tmpdir}/"

        out_annotate_file = annotate_bed(
            input_bed_file,
            intersection_cutoff,
            coverage_lcr_file,
            prefix,
            min_cnv_length,
        )
        assert filecmp.cmp(out_annotate_file, expected_out_annotate_bed_file)

    def test_aggregate_annotations_in_df(self, tmpdir):
        """Test aggregating coverage annotations into a DataFrame."""
        # Create a primary bed file with various annotation formats
        primary_bed_content = (
            "chr1\t1000\t2000\tCN2\n" "chr1\t3000\t4000\tCN3|UG-CNV-LCR\n" "chr2\t5000\t6000\tCN1|LEN;CN1|UG-CNV-LCR\n"
        )
        primary_bed_file = tmpdir.join("primary.bed")
        primary_bed_file.write(primary_bed_content)

        # Create coverage annotation files
        sample_mean_content = (
            "chr1\t1000\t2000\tregion1\t1.5\n" "chr1\t3000\t4000\tregion2\t2.3\n" "chr2\t5000\t6000\tregion3\t0.8\n"
        )
        sample_mean_file = tmpdir.join("cov_mean.bed")
        sample_mean_file.write(sample_mean_content)

        sample_std_content = (
            "chr1\t1000\t2000\tregion1\t0.1\n" "chr1\t3000\t4000\tregion2\t0.2\n" "chr2\t5000\t6000\tregion3\t0.15\n"
        )
        sample_std_file = tmpdir.join("cov_std.bed")
        sample_std_file.write(sample_std_content)

        cohort_mean_content = (
            "chr1\t1000\t2000\tregion1\t1.0\n" "chr1\t3000\t4000\tregion2\t1.0\n" "chr2\t5000\t6000\tregion3\t1.0\n"
        )
        cohort_mean_file = tmpdir.join("cohort_mean.bed")
        cohort_mean_file.write(cohort_mean_content)

        cohort_std_content = (
            "chr1\t1000\t2000\tregion1\t0.05\n" "chr1\t3000\t4000\tregion2\t0.06\n" "chr2\t5000\t6000\tregion3\t0.07\n"
        )
        cohort_std_file = tmpdir.join("cohort_std.bed")
        cohort_std_file.write(cohort_std_content)

        # Prepare coverage annotations list
        coverage_annotations = [
            ("sample", "mean", str(sample_mean_file)),
            ("sample", "stdev", str(sample_std_file)),
            ("cohort", "mean", str(cohort_mean_file)),
            ("cohort", "stdev", str(cohort_std_file)),
        ]

        # Call the function
        result_df = aggregate_annotations_in_df(str(primary_bed_file), coverage_annotations)

        # Verify the structure
        assert list(result_df.columns) == [
            "chr",
            "start",
            "end",
            "CopyNumber",
            "filter",
            "SVTYPE",
            "CNMOPS_SAMPLE_MEAN",
            "CNMOPS_SAMPLE_STDEV",
            "CNMOPS_COHORT_MEAN",
            "CNMOPS_COHORT_STDEV",
        ]

        # Verify the number of rows
        assert len(result_df) == 3

        # Verify the data for first row (no filters)
        assert result_df.iloc[0]["chr"] == "chr1"
        assert result_df.iloc[0]["start"] == 1000
        assert result_df.iloc[0]["end"] == 2000
        assert result_df.iloc[0]["CopyNumber"] == 2
        assert result_df.iloc[0]["filter"] == "PASS"  # PASS string when no filters
        assert result_df.iloc[0]["SVTYPE"] == "NEUTRAL"  # CN2 is neutral
        assert result_df.iloc[0]["CNMOPS_SAMPLE_MEAN"] == 1.5
        assert result_df.iloc[0]["CNMOPS_SAMPLE_STDEV"] == 0.1
        assert result_df.iloc[0]["CNMOPS_COHORT_MEAN"] == 1.0
        assert result_df.iloc[0]["CNMOPS_COHORT_STDEV"] == 0.05

        # Verify second row (single filter)
        assert result_df.iloc[1]["CopyNumber"] == 3  # Integer, not "CN3"
        assert result_df.iloc[1]["filter"] == "UG-CNV-LCR"  # String with one filter
        assert result_df.iloc[1]["SVTYPE"] == "DUP"  # CN3 is duplication
        assert result_df.iloc[1]["CNMOPS_SAMPLE_MEAN"] == 2.3

        # Verify third row (semicolon-separated with multiple filters)
        assert result_df.iloc[2]["CopyNumber"] == 1  # Integer, not "CN1"
        assert result_df.iloc[2]["filter"] == "LEN,UG-CNV-LCR"  # Comma-separated string with two filters
        assert result_df.iloc[2]["SVTYPE"] == "DEL"  # CN1 is deletion
        assert result_df.iloc[2]["CNMOPS_SAMPLE_MEAN"] == 0.8

    def test_aggregate_annotations_empty_coverage_list(self, tmpdir):
        """Test aggregating with empty coverage annotations list."""
        # Create a primary bed file
        primary_bed_content = "chr1\t1000\t2000\tCN2\n" "chr1\t3000\t4000\tCN3|UG-CNV-LCR\n"
        primary_bed_file = tmpdir.join("primary.bed")
        primary_bed_file.write(primary_bed_content)

        # Call with empty coverage annotations
        result_df = aggregate_annotations_in_df(str(primary_bed_file), [])

        # Verify the structure - should only have the basic columns
        assert list(result_df.columns) == ["chr", "start", "end", "CopyNumber", "filter", "SVTYPE"]

        # Verify the data
        assert len(result_df) == 2
        assert result_df.iloc[0]["CopyNumber"] == 2  # Integer
        assert result_df.iloc[0]["filter"] == "PASS"  # PASS string when no filters
        assert result_df.iloc[0]["SVTYPE"] == "NEUTRAL"  # CN2 is neutral
        assert result_df.iloc[1]["CopyNumber"] == 3  # Integer
        assert result_df.iloc[1]["filter"] == "UG-CNV-LCR"  # String
        assert result_df.iloc[1]["SVTYPE"] == "DUP"  # CN3 is duplication
