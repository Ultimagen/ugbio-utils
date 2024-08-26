import os
import warnings
from pathlib import Path
import pytest

warnings.filterwarnings('ignore')

from ugbio_cnv import plot_FREEC_fold_change


@pytest.fixture
def resources_dir():
    return Path(__file__).parent / "resources"


class TestPlotFREECFoldChange:

    def test_plot_FREEC_fold_change(self, tmpdir, resources_dir):
        input_ratio_file = resources_dir / "038266-HG008_T-Z0137-CTTCATGCATCTCAGAT.cram.CHR19_ratio.txt"
        expected_outfile = resources_dir / "HG008.fold_change.jpeg"

        sample_name = 'test_sample'
        out_dir = Path(tmpdir)
        df = plot_FREEC_fold_change.read_ratio_file(input_ratio_file)
        out_file = plot_FREEC_fold_change.plot_ratio_values(df, sample_name, out_dir)


        out_file = Path(tmpdir) / f"{sample_name}.fold_change.jpeg"

        assert os.path.getsize(out_file)>0
        
