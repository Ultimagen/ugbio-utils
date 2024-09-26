from os.path import join as pjoin
from pathlib import Path

import pytest
from ugbio_core import intersect_bed_regions

@pytest.fixture
def resources_dir():
    return Path(__file__).parent / "resources"

def test_intersect_bed_regions(tmpdir, resources_dir):
    intersect_bed_regions.run(
        [
            "intersect_bed_regions",
            "--include-regions",
            pjoin(resources_dir, "arbitrary_region.chr20.bed"),
            pjoin(resources_dir, "ug_hcr.chr20.subset.bed"),
            "--exclude-regions",
            pjoin(resources_dir, "arbitrary_exclude_region.chr20.bed"),
            pjoin(resources_dir, "Homo_sapiens_assembly38.dbsnp138.chr20_subset.vcf.gz"),
            "--output-bed",
            pjoin(tmpdir, "output.bed"),
        ]
    )

    with open(pjoin(tmpdir, "output.bed"), "r") as f:
        result = f.readlines()
    with open(pjoin(resources_dir, "expected_output.bed"), "r") as f:
        expected_result = f.readlines()

    assert result == expected_result
