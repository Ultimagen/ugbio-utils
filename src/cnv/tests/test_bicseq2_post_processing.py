import filecmp
from os.path import join as pjoin
from pathlib import Path

import pytest
from ugbio_cnv import bicseq2_post_processing


@pytest.fixture
def resources_dir():
    return Path(__file__).parent / "resources"


class TestBicseq2PostProcessing:

    def test_bicseq2_post_processing(self, tmpdir, resources_dir):
        input_bicseq2_txt_file = pjoin(resources_dir, "T_N_HCC1143_CHR22_test.bicseq2.txt")
        expected_out_bed_file = pjoin(resources_dir, "expected_T_N_HCC1143_CHR22_test.bicseq2.bed")
        prefix = f"{tmpdir}/"
        out_file = f"{tmpdir}/T_N_HCC1143_CHR22_test.bicseq2.bed"

        bicseq2_post_processing.run([
            "bicseq2_post_processing",
            "--input_bicseq2_txt_file",
            input_bicseq2_txt_file,
            "--out_directory",
            prefix
        ])
        assert filecmp.cmp(expected_out_bed_file, out_file)
