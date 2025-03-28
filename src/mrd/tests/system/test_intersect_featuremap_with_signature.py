import filecmp
from os.path import join as pjoin
from pathlib import Path

import pytest
from ugbio_mrd.mrd_utils import intersect_featuremap_with_signature


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def test_intersect_featuremap_with_signature(tmpdir, resources_dir):
    signature = pjoin(
        resources_dir,
        "150382-BC04.filtered_signature.chr22_12693463.vcf.gz",
    )
    featuremap = pjoin(
        resources_dir,
        "featuremap_150419-BC04.sorted.chr22_12693463.vcf.gz",
    )
    expected_intersection = pjoin(
        resources_dir,
        "featuremap_150419-BC04.sorted.chr22_12693463.intersection.vcf.gz",
    )
    output_intersection = pjoin(tmpdir, "intersection.vcf.gz")
    intersect_featuremap_with_signature(
        featuremap_file=featuremap,
        signature_file=signature,
        output_intersection_file=output_intersection,
    )
    filecmp.cmp(output_intersection, expected_intersection)
