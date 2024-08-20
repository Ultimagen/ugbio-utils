# import pytest
import filecmp
import os
from os.path import join as pjoin
from pathlib import Path
from ugbio_core.pileuptofreq import create_frequncies_from_pileup


# @pytest.fixture
# def resources_dir():
#     return Path(__file__).parent / "resources"

def test_pileuptofreq(tmpdir):
    current_dir = os.path.dirname(__file__)
    input_pileup_file = pjoin(current_dir,'resources',"tumor.031865-Lb_2211-Z0048-CTGCCAGACTGTGAT.cram_minipileup.pileup")
    expected_outfile = pjoin(current_dir,'resources',"tumor.031865-Lb_2211-Z0048-CTGCCAGACTGTGAT.cram_minipileup.pileup.freq")
    df_freq = create_frequncies_from_pileup(input_pileup_file)
    outfile = pjoin(tmpdir,"out.pileup.freq")
    df_freq.to_csv(outfile,sep='\t',index=None)
    
    assert filecmp.cmp(outfile, expected_outfile)