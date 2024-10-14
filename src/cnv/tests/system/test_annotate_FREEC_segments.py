import filecmp
import os
import warnings
from os.path import join as pjoin
from pathlib import Path

import pytest

warnings.filterwarnings('ignore')

from ugbio_cnv import annotate_FREEC_segments


@pytest.fixture
def resources_dir():
    return Path(__file__).parent / "resources"


class TestAnnotateFREECSegments:

    def test_annotate_FREEC_segments(self, tmpdir, resources_dir):
        input_segments_file = pjoin(resources_dir, "in_segments.txt")
        gain_cutoff = 1.03
        loss_cutoff = 0.97
        expected_out_segments_annotated = pjoin(resources_dir, 'expected_in_segments_annotated.txt')
        expected_out_segments_CNVs = pjoin(resources_dir, 'expected_in_segments_CNVs.bed')

        annotate_FREEC_segments.run([
            "annotate_FREEC_segments",
            "--input_segments_file",
            input_segments_file,
            "--gain_cutoff",
            str(gain_cutoff),
            "--loss_cutoff",
            str(loss_cutoff),
        ])

        out_segments_annotated = os.path.basename(input_segments_file) + '_annotated.txt'
        out_segments_CNVs = os.path.basename(input_segments_file) + '_CNVs.bed'
        assert filecmp.cmp(out_segments_annotated, expected_out_segments_annotated)
        assert filecmp.cmp(out_segments_CNVs, expected_out_segments_CNVs)
