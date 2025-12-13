import os
from os.path import dirname
from pathlib import Path

import pytest
from ugbio_comparison import sv_comparison_pipeline


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


class TestSVComparisonPipeline:
    def test_sv_comparison_pipeline(self, tmpdir, resources_dir):
        output_file = f"{tmpdir}/HG002.h5"
        os.makedirs(dirname(output_file), exist_ok=True)
        sv_comparison_pipeline.run(
            [
                "sv_comparison_pipeline",
                "--calls",
                f"{resources_dir}/hg002.ug.release.chr9.vcf.gz",
                "--gt",
                f"{resources_dir}/GRCh38_HG2-T2TQ100-V1.1_stvar.chr9.vcf.gz",
                "--hcr_bed",
                f"{resources_dir}/GRCh38_HG2-T2TQ100-V1.1_stvar.benchmark.chr9.bed",
                "--output_filename",
                output_file,
                "--outdir",
                os.path.join(str(tmpdir), "truvari"),
            ]
        )
        assert os.path.exists(output_file)

    def test_sv_comparison_pipeline_nofilter(self, tmpdir, resources_dir):
        output_file = f"{tmpdir}/HG002.h5"
        os.makedirs(dirname(output_file), exist_ok=True)
        sv_comparison_pipeline.run(
            [
                "sv_comparison_pipeline",
                "--calls",
                f"{resources_dir}/hg002.ug.release.chr9.vcf.gz",
                "--gt",
                f"{resources_dir}/GRCh38_HG2-T2TQ100-V1.1_stvar.chr9.vcf.gz",
                "--hcr_bed",
                f"{resources_dir}/GRCh38_HG2-T2TQ100-V1.1_stvar.benchmark.chr9.bed",
                "--output_filename",
                output_file,
                "--outdir",
                os.path.join(str(tmpdir), "truvari"),
                "--ignore_filter",
            ]
        )
        assert os.path.exists(output_file)

    def test_sv_comparison_pipeline_skip_collapse(self, tmpdir, resources_dir):
        output_file = f"{tmpdir}/HG002_skip_collapse.h5"
        os.makedirs(dirname(output_file), exist_ok=True)
        sv_comparison_pipeline.run(
            [
                "sv_comparison_pipeline",
                "--calls",
                f"{resources_dir}/hg002.ug.release.chr9.vcf.gz",
                "--gt",
                f"{resources_dir}/GRCh38_HG2-T2TQ100-V1.1_stvar.chr9.vcf.gz",
                "--hcr_bed",
                f"{resources_dir}/GRCh38_HG2-T2TQ100-V1.1_stvar.benchmark.chr9.bed",
                "--output_filename",
                output_file,
                "--outdir",
                os.path.join(str(tmpdir), "truvari_skip_collapse"),
                "--skip_collapse",
            ]
        )
        assert os.path.exists(output_file)
        # Verify that ground truth collapsed files exist but calls collapsed files don't
        truvari_dir = os.path.join(str(tmpdir), "truvari_skip_collapse")
        gt_files = [f for f in os.listdir(truvari_dir) if "GRCh38_HG2" in f and "collapsed" in f]
        calls_files = [f for f in os.listdir(truvari_dir) if "hg002.ug" in f and "collapsed" in f]
        assert len(gt_files) > 0, "Ground truth should have collapsed files"
        assert len(calls_files) == 0, "Calls should not have collapsed files when skip_collapse=True"
