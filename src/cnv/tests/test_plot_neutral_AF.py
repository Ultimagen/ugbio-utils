import os
from os.path import join as pjoin
import filecmp
from pathlib import Path
import pytest

@pytest.fixture
def resources_dir():
    return Path(__file__).parent / "resources"

from ugbio_cnv import plot_FREEC_neutral_AF

class TestPlotFREECNeutralAF:
    def test_plot_FREEC_neutral_AF(self, tmpdir, resources_dir):
        input_cnv_file = pjoin(resources_dir, "COLO829.full_sample.sorter_input.test.cnvs.filter.CHR19.bed")
        input_mpileup = pjoin(resources_dir, "tumor.031865-Lb_2211-Z0048-CTGCCAGACTGTGAT.cram_minipileup.CHR19.pileup")

        expected_AF_bed_file = pjoin(resources_dir,'expected_COLO829_CHR19.freq.SNP.neutral.bed')
        expected_AF_hist_fig = pjoin(resources_dir,'expected_COLO829_CHR19.freq.SNP.neutral.hist.jpeg')        
        
        sample_name = 'COLO829_CHR19'
        out_dir = f"{tmpdir}"
        plot_FREEC_neutral_AF.run([
            "plot_FREEC_neutral_AF",
            "--mpileup", 
            input_mpileup,
            "--cnvs_file",
            input_cnv_file,
            "--sample_name",
            sample_name,
            "--out_directory", 
            out_dir
            ])
        
        basename = os.path.basename(input_mpileup)
        out_AF_hist_fig = f"{tmpdir}/{basename}.freq.SNP.neutral.hist.jpeg"
        out_AF_bed_file = f"{tmpdir}/{basename}.freq.SNP.neutral.bed"

        assert os.path.getsize(out_AF_hist_fig)>0
        assert filecmp.cmp(out_AF_bed_file, expected_AF_bed_file)
