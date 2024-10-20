import filecmp
import os
from pathlib import Path

import pytest
from ugbio_cnv import plot_FREEC_neutral_AF


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


class TestPlotFREECNeutralAF:
    def test_plot_freec_neutral_af(self, tmpdir, resources_dir):
        input_cnv_file = str(resources_dir / "COLO829.full_sample.sorter_input.test.cnvs.filter.CHR19.bed")
        input_mpileup = str(resources_dir / "tumor.031865-Lb_2211-Z0048-CTGCCAGACTGTGAT.cram_minipileup.CHR19.pileup")

        expected_af_bed_file = str(resources_dir / "expected_COLO829_CHR19.freq.SNP.neutral.bed")
        # expected_af_hist_fig = str(resources_dir / "expected_COLO829_CHR19.freq.SNP.neutral.hist.jpeg")

        sample_name = "COLO829_CHR19"
        out_dir = f"{tmpdir}"
        plot_FREEC_neutral_AF.run(
            [
                "plot_FREEC_neutral_AF",
                "--mpileup",
                input_mpileup,
                "--cnvs_file",
                input_cnv_file,
                "--sample_name",
                sample_name,
                "--out_directory",
                out_dir,
            ]
        )

        basename = os.path.basename(input_mpileup)
        out_af_hist_fig = Path(tmpdir) / f"{basename}.freq.SNP.neutral.hist.jpeg"
        out_af_bed_file = Path(tmpdir) / f"{basename}.freq.SNP.neutral.bed"

        assert os.path.getsize(out_af_hist_fig) > 0
        assert filecmp.cmp(out_af_bed_file, expected_af_bed_file)
