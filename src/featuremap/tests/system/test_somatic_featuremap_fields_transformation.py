import os
from collections import defaultdict
from os.path import join as pjoin
from pathlib import Path

import pysam
import pytest
from ugbio_featuremap import somatic_featuremap_fields_transformation


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


class TestSomaticFeaturemapFieldsTransformation:
    def test_somatic_featuremap_fields_transformation(self, tmpdir, resources_dir):
        somatic_featuremap_vcf = pjoin(resources_dir, "Pa_46.tumor_normal.merged.tumor_PASS.mpileup_PASS.chr19.vcf.gz")
        ref_tr_file = pjoin(resources_dir, "tr_hg38.chr19.bed")
        interval_list = pjoin(resources_dir, "wgs_calling_regions.hg38.chr19_test.interval_list")
        filter_string = "PASS"
        out_dir = tmpdir
        out_file = pjoin(out_dir, "Pa_46.tumor_normal.merged.tumor_PASS.mpileup_PASS.chr19.xgb_proba.vcf.gz")

        expected_num_variants = 359

        # Run the script's main function
        somatic_featuremap_fields_transformation.run(
            [
                "somatic_featuremap_fields_transformation",
                "-sfm",
                somatic_featuremap_vcf,
                "-o",
                out_file,
                "-i",
                interval_list,
                "-filter_string",
                filter_string,
                "-ref_tr",
                ref_tr_file,
            ]
        )

        # check that the output file exists and has the expected content
        assert os.path.isfile(out_file)
        # count the number of variants (excluding the header)
        cons_dict = defaultdict(dict)
        for rec in pysam.VariantFile(out_file):
            rec_id = (rec.chrom, rec.pos, rec.ref, rec.alts[0])
            if rec_id not in cons_dict:
                cons_dict[rec_id]["count"] = 0
            cons_dict[rec_id]["count"] += 1
        # check the number of variants
        num_variants = len(cons_dict)
        assert num_variants == expected_num_variants
        # assert a single entry per variant
        assert all(cons["count"] == 1 for cons in cons_dict.values())

        # Check header has expected INFO and FORMAT fields
        expected_info_fields = (
            list(somatic_featuremap_fields_transformation.added_info_features.keys())
            + somatic_featuremap_fields_transformation.info_fields_for_training
        )
        expected_format_fields = somatic_featuremap_fields_transformation.added_format_features

        vcf_reader = pysam.VariantFile(out_file)
        # Check all expected INFO fields are present in header
        for field in expected_info_fields:
            assert field in vcf_reader.header.info, f"Missing INFO field: {field}"
        for field in expected_format_fields:
            assert field in vcf_reader.header.formats, f"Missing FORMAT field: {field}"

        # Check that records contain the expected fields
        for rec in vcf_reader:
            # Check INFO fields in records
            for field in expected_info_fields:
                assert field in rec.info, f"Record missing INFO field: {field}"
            for field in expected_format_fields:
                assert field in rec.samples[0], f"Sample[0] missing FORMAT field: {field}"

        vcf_reader.close()
