import os
import warnings
from os.path import join as pjoin
from pathlib import Path

import pytest
from ugbio_cnv import plot_cnv_results

warnings.filterwarnings("ignore")


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


class TestPlotCnvResults:
    def test_plot_germline_cnv_results(self, tmpdir, resources_dir):
        """
        Expected figures can be found under test resources:
        - expected_NA11428_cnv_figures/NA11428.chr1_chr2.CNV.calls.jpeg
        - expected_NA11428_cnv_figures/NA11428.chr1_chr2.CNV.coverage.jpeg
        - expected_NA11428_cnv_figures/NA11428.chr1_chr2.dup_del.calls.jpeg
        """
        input_germline_coverage = pjoin(resources_dir, "NA11428.chr1_chr2.ReadCounts.bed")
        input_dup_bed = pjoin(resources_dir, "NA11428.chr1_chr2.DUP.cnvs.filter.bed")
        input_del_bed = pjoin(resources_dir, "NA11428.chr1_chr2.DEL.cnvs.filter.bed")

        sample_name = "NA11428.chr1_chr2"
        out_dir = f"{tmpdir}"
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
                input_dup_bed,
                "--deletion_cnv_calls",
                input_del_bed,
                "--vcf-like",
            ]
        )

        out_calls_fig = f"{tmpdir}/{sample_name}.CNV.calls.jpeg"
        out_coverage_fig = f"{tmpdir}/{sample_name}.CNV.coverage.jpeg"
        out_dup_del_fig = f"{tmpdir}/{sample_name}.dup_del.calls.jpeg"

        assert os.path.getsize(out_calls_fig) > 0
        assert os.path.getsize(out_coverage_fig) > 0
        assert os.path.getsize(out_dup_del_fig) > 0

    def test_plot_somatic_cnv_results(self, tmpdir, resources_dir):
        """
        Expected figures can be found under test resources:
        - somatic/expected_figures/COLO829.chr1_chr2.CNV.calls.jpeg
        - somatic/expected_figures/COLO829.chr1_chr2.CNV.coverage.jpeg
        - somatic/expected_figures/COLO829.chr1_chr2.dup_del.calls.jpeg
        """
        input_germline_coverage = pjoin(resources_dir, "somatic", "germline.bedGraph")
        input_tumor_coverage = pjoin(resources_dir, "somatic", "tumor.bedGraph")
        input_dup_bed = pjoin(resources_dir, "somatic", "COLO829_run031865.cnvs.filter.chr1_chr2.DUP.bed")
        input_del_bed = pjoin(resources_dir, "somatic", "COLO829_run031865.cnvs.filter.chr1_chr2.DEL.bed")
        input_gt_dup_bed = pjoin(resources_dir, "somatic", "COLO-829.GT.chr1_chr2.DUP.bed")
        input_gt_del_bed = pjoin(resources_dir, "somatic", "COLO-829.GT.chr1_chr2.DEL.bed")

        sample_name = "COLO829.chr1_chr2"
        out_dir = f"{tmpdir}"
        plot_cnv_results.run(
            [
                "plot_cnv_results",
                "--germline_coverage",
                input_germline_coverage,
                "--tumor_coverage",
                input_tumor_coverage,
                "--duplication_cnv_calls",
                input_dup_bed,
                "--deletion_cnv_calls",
                input_del_bed,
                "--gt_duplication_cnv_calls",
                input_gt_dup_bed,
                "--gt_deletion_cnv_calls",
                input_gt_del_bed,
                "--out_directory",
                out_dir,
                "--sample_name",
                sample_name,
            ]
        )

        out_calls_fig = f"{tmpdir}/{sample_name}.CNV.calls.jpeg"
        out_coverage_fig = f"{tmpdir}/{sample_name}.CNV.coverage.jpeg"
        out_dup_del_fig = f"{tmpdir}/{sample_name}.dup_del.calls.jpeg"

        assert os.path.getsize(out_calls_fig) > 0
        assert os.path.getsize(out_coverage_fig) > 0
        assert os.path.getsize(out_dup_del_fig) > 0

    def test_plot_cnv_with_vcf_like_format_copynumber_tag(self, tmpdir, resources_dir):
        # Create a temporary CNV calls file with VCF-like INFO format
        dup_content = (
            "chr1\t1000000\t1500000\tSVTYPE=DUP;CopyNumber=3.5;END=1500000\n"
            "chr1\t2000000\t2500000\tSVTYPE=DUP;CopyNumber=4.2;QUAL=HIGH\n"
        )

        dup_file = tmpdir.join("test_dup.bed")
        dup_file.write(dup_content)

        input_germline_coverage = pjoin(resources_dir, "NA11428.chr1_chr2.ReadCounts.bed")
        sample_name = "test_copynumber_tag"
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
                str(dup_file),
                "--vcf-like",
            ]
        )

        out_calls_fig = tmpdir.join(f"{sample_name}.CNV.calls.jpeg")
        out_coverage_fig = tmpdir.join(f"{sample_name}.CNV.coverage.jpeg")
        out_dup_del_fig = tmpdir.join(f"{sample_name}.dup_del.calls.jpeg")

        assert out_calls_fig.size() > 0
        assert out_coverage_fig.size() > 0
        assert out_dup_del_fig.size() > 0

    def test_plot_cnv_with_vcf_like_format_cn_tag(self, tmpdir, resources_dir):
        # Create a temporary CNV calls file with CN= tag only
        del_content = (
            "chr1\t3000000\t3500000\tSVTYPE=DEL;CN=1;END=3500000\n"
            "chr2\t4000000\t4500000\tSVTYPE=DEL;CN=0.5;QUAL=30\n"
        )

        del_file = tmpdir.join("test_del.bed")
        del_file.write(del_content)

        input_germline_coverage = pjoin(resources_dir, "NA11428.chr1_chr2.ReadCounts.bed")
        sample_name = "test_cn_tag"
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
                "--deletion_cnv_calls",
                str(del_file),
                "--vcf-like",
            ]
        )

        out_calls_fig = tmpdir.join(f"{sample_name}.CNV.calls.jpeg")
        out_coverage_fig = tmpdir.join(f"{sample_name}.CNV.coverage.jpeg")
        out_dup_del_fig = tmpdir.join(f"{sample_name}.dup_del.calls.jpeg")

        assert out_calls_fig.size() > 0
        assert out_coverage_fig.size() > 0
        assert out_dup_del_fig.size() > 0
