import filecmp
from pathlib import Path

import pytest
from ugbio_freec import bigwig_to_bedgraph


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def test_featuremap_to_dataframe(tmpdir, resources_dir):
    bigwig_file = resources_dir / "test_input.bigWig"
    bedgraph_expected_file = resources_dir / "texpected_output.bedGraph"
    bedgraph_actual_file = tmpdir / "output.bedGraph"
    bigwig_to_bedgraph(bigwig_file, bedgraph_actual_file)

    assert filecmp.cmp(bedgraph_expected_file, bedgraph_actual_file)
