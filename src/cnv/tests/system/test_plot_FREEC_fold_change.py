import os
import warnings
from pathlib import Path

import pytest
from ugbio_cnv import plot_FREEC_fold_change

warnings.filterwarnings("ignore")


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


class TestPlotFREECFoldChange:
    def test_plot_freec_fold_change(self, tmpdir, resources_dir):
        """
        Expected figures can be found under test resources: HG008.fold_change.jpeg"""
        input_ratio_file = resources_dir / "038266-HG008_T-Z0137-CTTCATGCATCTCAGAT.cram.CHR19_ratio.txt"

        sample_name = "test_sample"
        out_dir = Path(tmpdir)
        ratio_df = plot_FREEC_fold_change.read_ratio_file(input_ratio_file)
        out_file = plot_FREEC_fold_change.plot_ratio_values(ratio_df, sample_name, out_dir)

        out_file = Path(tmpdir) / f"{sample_name}.fold_change.jpeg"

        assert os.path.getsize(out_file) > 0
