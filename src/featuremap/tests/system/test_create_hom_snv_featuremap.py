from collections import defaultdict
from os.path import join as pjoin
from pathlib import Path

import pysam
import pytest
from ugbio_featuremap import create_hom_snv_featuremap


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def test_create_hom_snv_featuremap(tmpdir, resources_dir):
    # this vcf spans 5 HOM SNV loci, out of which 3 meet the criteria, 1 is low coverage (10) and 1 is low AF (9/30)
    hom_snv_featuremap = pjoin(tmpdir, "Pa_46_333_LuNgs_08.chr1_1000291_1002833.hom_snv.vcf.gz")
    create_hom_snv_featuremap.run(
        [
            "create_hom_snv_featuremap.py",
            "--featuremap",
            pjoin(resources_dir, "Pa_46_333_LuNgs_08.chr1_1000291_1002833.vcf.gz"),
            "--sorter_stats_json",
            pjoin(resources_dir, "Pa_46_333_LuNgs_08.json"),
            "--hom_snv_featuremap",
            hom_snv_featuremap,
            "--requested_min_coverage",
            "20",
            "--min_af",
            "0.7",
        ]
    )

    # making sure the output is as expected, with the 3 correct loci
    loci_counter = defaultdict(int)
    with pysam.VariantFile(hom_snv_featuremap) as fvcf:
        for record in fvcf.fetch():
            loci_counter[(record.chrom, record.pos)] += 1
    assert loci_counter[("chr1", 1002308)] == 59
    assert loci_counter[("chr1", 1002736)] == 56
    assert loci_counter[("chr1", 1002833)] == 54
    assert len(loci_counter) == 3


def test_create_hom_snv_featuremap_with_similar_call(tmpdir, resources_dir):
    # this vcf spans 5 HOM SNV loci, out of which 3 meet the criteria, 1 is low coverage (10) and 1 is low AF (9/30)
    hom_snv_featuremap = pjoin(tmpdir, "Pa_46_333_LuNgs_08.chr1_1000291_1002833.hom_snv.with.similar.vcf.gz")
    create_hom_snv_featuremap.run(
        [
            "create_hom_snv_featuremap.py",
            "--featuremap",
            pjoin(resources_dir, "Pa_46_333_LuNgs_08.chr1_1000291_1002833.with.similar.vcf.gz"),
            "--sorter_stats_json",
            pjoin(resources_dir, "Pa_46_333_LuNgs_08.json"),
            "--hom_snv_featuremap",
            hom_snv_featuremap,
            "--requested_min_coverage",
            "20",
            "--min_af",
            "0.7",
        ]
    )

    # making sure the output is as expected, with the 3 correct loci
    loci_counter = defaultdict(int)
    with pysam.VariantFile(hom_snv_featuremap) as fvcf:
        for record in fvcf.fetch():
            loci_counter[(record.chrom, record.pos)] += 1
    assert loci_counter[("chr1", 1002308)] == 59
    assert loci_counter[("chr1", 1002736)] == 55
    assert loci_counter[("chr1", 1002833)] == 54
    assert len(loci_counter) == 3
