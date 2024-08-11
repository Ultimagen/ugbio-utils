import filecmp
from os.path import join as pjoin
from pathlib import Path

import pytest
from ugbio_cnv.filter_sample_cnvs import annotate_bed


@pytest.fixture
def resources_dir():
    return Path(__file__).parent / "resources"


class TestFilterSampleCnvs:

    def test_annotate_bed(self, tmpdir, resources_dir):
        input_bed_file = pjoin(resources_dir, "unfiltered_cnvs.bed")
        expected_out_filtered_bed_file = pjoin(resources_dir, "filtered_cnvs.bed")
        expected_out_annotate_bed_file = pjoin(resources_dir, "annotate_cnv.bed")
        coverage_lcr_file = pjoin(resources_dir, "UG-CNV-LCR.bed")
        intersection_cutoff = 0.5
        min_cnv_length = 10000
        prefix = f"{tmpdir}/"

        [out_annotate_file, out_filtered_file] = annotate_bed(
            input_bed_file,
            intersection_cutoff,
            coverage_lcr_file,
            prefix,
            min_cnv_length,
        )
        print(out_filtered_file)
        assert filecmp.cmp(out_filtered_file, expected_out_filtered_bed_file)
        assert filecmp.cmp(out_annotate_file, expected_out_annotate_bed_file)
