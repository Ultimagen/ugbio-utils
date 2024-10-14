import filecmp
from os.path import basename, exists
from pathlib import Path

import pytest
from ugbio_core import trimmer_utils


@pytest.fixture
def inputs_dir():
    inputs_dir = Path(__file__).parent / "resources" 
    return inputs_dir

def test_merge_trimmer_histograms(tmpdir, inputs_dir):
    trimmer_histograms = [
        str(inputs_dir / f"029917001_1_Z0098.histogram.csv"),
        str(inputs_dir / f"029917001_2_Z0098.histogram.csv")
        ]
    expected_output = str(inputs_dir / f"EXPECTED.029917001_Z0098.histogram.csv")
    output_path = str (Path(tmpdir) / basename(trimmer_histograms[0]))

    merged_histogram = trimmer_utils.merge_trimmer_histograms(trimmer_histograms=trimmer_histograms, output_path=tmpdir)
    assert exists(merged_histogram)
    assert merged_histogram == output_path
    assert filecmp.cmp(merged_histogram, expected_output)

    output_path = str (Path(tmpdir) / basename(trimmer_histograms[1]))
    merged_histogram = trimmer_utils.merge_trimmer_histograms(
        trimmer_histograms=trimmer_histograms[1], output_path=tmpdir
    )
    assert exists(merged_histogram)
    assert filecmp.cmp(merged_histogram, trimmer_histograms[1])