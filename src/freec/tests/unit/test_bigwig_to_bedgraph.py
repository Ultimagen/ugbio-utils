import filecmp
from os.path import join as pjoin
from pathlib import Path

import pyBigWig as pBW
import pytest
from ugbio_freec.bigwig_to_bedgraph import bigwig_to_bedgraph


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def create_temp_bw(tmpdir, name):
    with pBW.open(pjoin(tmpdir, name), "w") as bw:
        bw.addHeader([("chr1", 1000)])
        bw.addEntries(
            ["chr1", "chr1", "chr1"],  # Chromosomes
            [0, 200, 500],  # Start positions
            ends=[100, 300, 600],  # End positions
            values=[1.0, 2.5, -3.0],  # Values
        )
    return pjoin(tmpdir, name)


def test_bigwig_to_bedgraph(tmpdir, resources_dir):
    tmp_bw_file = create_temp_bw(tmpdir, "test1.bw")
    bedgraph_expected_file = resources_dir / "expected_output.bedGraph"
    bedgraph_actual_file = tmpdir / "output.bedGraph"
    bigwig_to_bedgraph(tmp_bw_file, bedgraph_actual_file)

    assert filecmp.cmp(bedgraph_expected_file, bedgraph_actual_file)
