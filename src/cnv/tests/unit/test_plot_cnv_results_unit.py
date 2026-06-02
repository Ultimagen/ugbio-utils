"""
Unit tests for plot_cnv_results.py
Tests for BIOIN-2697: Ensure all three plots are generated and chromosome names are handled correctly.
"""

import os
from pathlib import Path

import pytest
from ugbio_cnv import plot_cnv_results


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


class TestPlotCnvResultsEmptyCalls:
    """Test that all three output files are generated even when CNV calls are empty (BIOIN-2697)."""

    def test_empty_cnv_calls_generates_all_plots(self, tmpdir, resources_dir):
        """
        Test Issue 1: When both DUP and DEL files are empty, all three plots should still be generated.
        Previously, plot_cnv_calls() would return None and skip the third plot.
        """
        # Use existing coverage file
        input_germline_coverage = os.path.join(resources_dir, "NA11428.chr1_chr2.ReadCounts.bed")

        # Create empty CNV call files
        empty_dup_file = tmpdir.join("empty.DUP.bed")
        empty_del_file = tmpdir.join("empty.DEL.bed")
        empty_dup_file.write("")
        empty_del_file.write("")

        sample_name = "test_empty_calls"
        out_dir = str(tmpdir)

        plot_cnv_results.run(
            [
                "plot_cnv_results",
                "--germline_coverage",
                input_germline_coverage,
                "--out_directory",
                out_dir,
                "--sample_name",
                sample_name,
                "--duplication_cnv_calls",
                str(empty_dup_file),
                "--deletion_cnv_calls",
                str(empty_del_file),
                "--vcf-like",
            ]
        )

        # Verify all three output files exist and are non-empty
        out_calls_fig = tmpdir.join(f"{sample_name}.CNV.calls.jpeg")
        out_coverage_fig = tmpdir.join(f"{sample_name}.CNV.coverage.jpeg")
        out_dup_del_fig = tmpdir.join(f"{sample_name}.dup_del.calls.jpeg")

        assert out_calls_fig.exists(), "CNV calls plot should be generated even with empty calls"
        assert out_coverage_fig.exists(), "Coverage plot should be generated"
        assert out_dup_del_fig.exists(), "Dup/Del plot should be generated"

        assert out_calls_fig.size() > 0, "CNV calls plot should not be empty"
        assert out_coverage_fig.size() > 0, "Coverage plot should not be empty"
        assert out_dup_del_fig.size() > 0, "Dup/Del plot should not be empty"


class TestPlotCnvResultsChromosomeNaming:
    """Test that chromosome names with and without 'chr' prefix are handled correctly (BIOIN-2697)."""

    def test_chromosome_names_without_chr_prefix(self, tmpdir):
        """
        Test Issue 2: Chromosome names without 'chr' prefix (e.g., '9' instead of 'chr9')
        should not cause AttributeError when using .str accessor.
        """
        # Create test coverage file with numeric chromosome names (no 'chr' prefix)
        cov_content = """9\t0\t1000\t50
9\t1000\t2000\t55
9\t2000\t3000\t48
9\t3000\t4000\t52
9\t4000\t5000\t49
9\t5000\t6000\t51
9\t6000\t7000\t53
9\t7000\t8000\t50
9\t8000\t9000\t54
9\t9000\t10000\t52
"""
        cov_file = tmpdir.join("test_chr9.cov.bed")
        cov_file.write(cov_content)

        # Create CNV calls file with numeric chromosome names
        dup_content = (
            "9\t1000\t5000\tEND=5000;CopyNumber=3;SVLEN=4000;SVTYPE=DUP;FILTER=PASS\n"
            "9\t7000\t9000\tEND=9000;CopyNumber=4;SVLEN=2000;SVTYPE=DUP;FILTER=PASS\n"
        )
        dup_file = tmpdir.join("test_chr9.DUP.bed")
        dup_file.write(dup_content)

        del_content = "9\t2500\t3500\tEND=3500;CopyNumber=1;SVLEN=1000;SVTYPE=DEL;FILTER=PASS\n"
        del_file = tmpdir.join("test_chr9.DEL.bed")
        del_file.write(del_content)

        sample_name = "test_numeric_chr"
        out_dir = str(tmpdir)

        # This should not raise AttributeError about .str accessor
        plot_cnv_results.run(
            [
                "plot_cnv_results",
                "--germline_coverage",
                str(cov_file),
                "--out_directory",
                out_dir,
                "--sample_name",
                sample_name,
                "--duplication_cnv_calls",
                str(dup_file),
                "--deletion_cnv_calls",
                str(del_file),
                "--vcf-like",
            ]
        )

        # Verify all three output files are generated
        out_calls_fig = tmpdir.join(f"{sample_name}.CNV.calls.jpeg")
        out_coverage_fig = tmpdir.join(f"{sample_name}.CNV.coverage.jpeg")
        out_dup_del_fig = tmpdir.join(f"{sample_name}.dup_del.calls.jpeg")

        assert out_calls_fig.exists(), "CNV calls plot should be generated with numeric chr names"
        assert out_coverage_fig.exists(), "Coverage plot should be generated with numeric chr names"
        assert out_dup_del_fig.exists(), "Dup/Del plot should be generated with numeric chr names"

        assert out_calls_fig.size() > 0
        assert out_coverage_fig.size() > 0
        assert out_dup_del_fig.size() > 0

    def test_chromosome_names_with_chr_prefix(self, tmpdir):
        """
        Test that chromosome names with 'chr' prefix still work correctly.
        This ensures backward compatibility.
        """
        # Create test coverage file with 'chr' prefix
        cov_content = """chr9\t0\t1000\t50
chr9\t1000\t2000\t55
chr9\t2000\t3000\t48
chr9\t3000\t4000\t52
chr9\t4000\t5000\t49
chr9\t5000\t6000\t51
chr9\t6000\t7000\t53
chr9\t7000\t8000\t50
chr9\t8000\t9000\t54
chr9\t9000\t10000\t52
"""
        cov_file = tmpdir.join("test_chr_prefix.cov.bed")
        cov_file.write(cov_content)

        # Create CNV calls file with 'chr' prefix
        dup_content = (
            "chr9\t1000\t5000\tEND=5000;CopyNumber=3;SVLEN=4000;SVTYPE=DUP;FILTER=PASS\n"
            "chr9\t7000\t9000\tEND=9000;CopyNumber=4;SVLEN=2000;SVTYPE=DUP;FILTER=PASS\n"
        )
        dup_file = tmpdir.join("test_chr_prefix.DUP.bed")
        dup_file.write(dup_content)

        empty_del_file = tmpdir.join("test_chr_prefix.DEL.bed")
        empty_del_file.write("")

        sample_name = "test_chr_prefix"
        out_dir = str(tmpdir)

        plot_cnv_results.run(
            [
                "plot_cnv_results",
                "--germline_coverage",
                str(cov_file),
                "--out_directory",
                out_dir,
                "--sample_name",
                sample_name,
                "--duplication_cnv_calls",
                str(dup_file),
                "--deletion_cnv_calls",
                str(empty_del_file),
                "--vcf-like",
            ]
        )

        # Verify all three output files are generated
        out_calls_fig = tmpdir.join(f"{sample_name}.CNV.calls.jpeg")
        out_coverage_fig = tmpdir.join(f"{sample_name}.CNV.coverage.jpeg")
        out_dup_del_fig = tmpdir.join(f"{sample_name}.dup_del.calls.jpeg")

        assert out_calls_fig.exists()
        assert out_coverage_fig.exists()
        assert out_dup_del_fig.exists()

        assert out_calls_fig.size() > 0
        assert out_coverage_fig.size() > 0
        assert out_dup_del_fig.size() > 0

    def test_chromosome_names_with_x_and_y(self, tmpdir):
        """
        Test that sex chromosomes (X, Y) are handled correctly with or without 'chr' prefix.
        """
        # Create test coverage file with chrX and chrY
        cov_content = """X\t0\t1000\t50
X\t1000\t2000\t55
X\t2000\t3000\t48
Y\t0\t1000\t30
Y\t1000\t2000\t32
Y\t2000\t3000\t28
"""
        cov_file = tmpdir.join("test_xy.cov.bed")
        cov_file.write(cov_content)

        empty_dup_file = tmpdir.join("test_xy.DUP.bed")
        empty_del_file = tmpdir.join("test_xy.DEL.bed")
        empty_dup_file.write("")
        empty_del_file.write("")

        sample_name = "test_xy"
        out_dir = str(tmpdir)

        # Should handle X and Y chromosomes correctly (converted to 23 and 24 for sorting)
        plot_cnv_results.run(
            [
                "plot_cnv_results",
                "--germline_coverage",
                str(cov_file),
                "--out_directory",
                out_dir,
                "--sample_name",
                sample_name,
                "--duplication_cnv_calls",
                str(empty_dup_file),
                "--deletion_cnv_calls",
                str(empty_del_file),
                "--vcf-like",
            ]
        )

        # Verify all three output files are generated
        out_calls_fig = tmpdir.join(f"{sample_name}.CNV.calls.jpeg")
        out_coverage_fig = tmpdir.join(f"{sample_name}.CNV.coverage.jpeg")
        out_dup_del_fig = tmpdir.join(f"{sample_name}.dup_del.calls.jpeg")

        assert out_calls_fig.exists()
        assert out_coverage_fig.exists()
        assert out_dup_del_fig.exists()

        assert out_calls_fig.size() > 0
        assert out_coverage_fig.size() > 0
        assert out_dup_del_fig.size() > 0
