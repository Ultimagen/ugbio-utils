from os.path import exists
from os.path import join as pjoin
from pathlib import Path
from unittest import mock
from unittest.mock import patch

import pytest
from simppl.simple_pipeline import SimplePipeline
from ugbio_core.vcf_utils import VcfUtils


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


@patch("subprocess.call")
def test_intersect_bed_files(mock_subprocess_call, tmp_path, resources_dir):
    bed1 = pjoin(resources_dir, "bed1.bed")
    bed2 = pjoin(resources_dir, "bed2.bed")
    output_path = pjoin(tmp_path, "output.bed")

    # Test with simple pipeline
    sp = SimplePipeline(0, 10)
    VcfUtils(sp).intersect_bed_files(bed1, bed2, output_path)

    VcfUtils().intersect_bed_files(bed1, bed2, output_path)
    mock_subprocess_call.assert_called_once_with(
        ["bedtools", "intersect", "-a", bed1, "-b", bed2], stdout=mock.ANY, shell=False
    )
    assert exists(output_path)
