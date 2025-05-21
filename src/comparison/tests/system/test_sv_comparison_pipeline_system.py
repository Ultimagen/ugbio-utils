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
                str(tmpdir),
            ]
        )
        assert os.path.exists(output_file)
