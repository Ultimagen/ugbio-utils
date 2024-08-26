import pytest
import filecmp
from pathlib import Path
from ugbio_core.pileuptofreq import create_frequncies_from_pileup


@pytest.fixture
def resources_dir():
    return Path(__file__).parent / "resources"

def test_pileuptofreq(tmpdir, resources_dir):
    input_pileup_file = resources_dir / "tumor.031865-Lb_2211-Z0048-CTGCCAGACTGTGAT.cram_minipileup.pileup"
    expected_outfile = resources_dir / "tumor.031865-Lb_2211-Z0048-CTGCCAGACTGTGAT.cram_minipileup.pileup.freq"
    df_freq = create_frequncies_from_pileup(input_pileup_file)
    outfile = Path(tmpdir) / "out.pileup.freq"
    df_freq.to_csv(outfile,sep='\t',index=None)
    
    assert filecmp.cmp(outfile, expected_outfile)