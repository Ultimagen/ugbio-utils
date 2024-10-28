from os.path import join as pjoin
from pathlib import Path

import pytest
from ugbio_core.sorter_stats_to_mean_coverage import run, sorter_stats_to_mean_coverage


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def test_sorter_stats_to_mean_coverage(tmpdir, resources_dir):
    input_json = pjoin(resources_dir, "Pa_394_Plasma.Lb_1597.runs_021144_021150_023049.json")
    expected_coverage = 196
    output_file = pjoin(tmpdir, "mean_coverage.txt")
    sorter_stats_to_mean_coverage(input_json, output_file)
    with open(output_file) as f:
        assert f.read() == f"{expected_coverage} "


def test_run_sorter_stats_to_mean_coverage(tmpdir, resources_dir):
    input_json = pjoin(resources_dir, "Pa_394_Plasma.Lb_1597.runs_021144_021150_023049.json")
    expected_coverage = 196
    output_file = pjoin(tmpdir, "mean_coverage.txt")
    run(
        [
            "sorter_stats_to_mean_coverage",
            "-i",
            input_json,
            "-o",
            output_file,
        ]
    )
    with open(output_file) as f:
        assert f.read() == f"{expected_coverage} "
