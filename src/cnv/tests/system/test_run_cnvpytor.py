import filecmp
import os
from pathlib import Path

import pytest
from ugbio_cnv import run_cnvpytor


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


class TestRunCNVpytor:
    def test_run_cnvpytor(self, tmp_path, resources_dir):
        input_cram_file = os.path.join(resources_dir, "HG002.chr19.s0.01.bam")
        ref_fasta = os.path.join(resources_dir, "chr19.fasta")
        sample_name = "test_HG002"
        bin_size = "500"
        out_dir = str(tmp_path)

        run_cnvpytor.run(
            [
                "run_cnvpytor",
                "--input_bam_cram_file",
                input_cram_file,
                "--sample_name",
                sample_name,
                "--ref_fasta",
                ref_fasta,
                "--bin_size",
                bin_size,
                "--chr_list",
                "chr19",
                "--out_directory",
                out_dir,
            ]
        )

        out_cnvs_file = os.path.join(tmp_path, f"{sample_name}.pytor.bin{bin_size}.CNVs.tsv")
        expected_out_cnvs_file = os.path.join(resources_dir, "HG002.pytor.bin500.CNVs.tsv")
        assert filecmp.cmp(out_cnvs_file, expected_out_cnvs_file)
