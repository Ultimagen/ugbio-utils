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
        interval_list_bed_file = pjoin(resources_dir, "wgs_calling_regions.hg38.chr19_test.interval_list.bed")
        genome_file = pjoin(resources_dir, "Homo_sapiens_assembly38.fasta.fai")
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
                interval_list_bed_file,
                "-filter_string",
                filter_string,
                "-ref_tr",
                ref_tr_file,
                "-g",
                genome_file,
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
        tr_info_fields = somatic_featuremap_fields_transformation.info_fields_for_training
        added_info_fields = list(somatic_featuremap_fields_transformation.added_info_features.keys())
        expected_format_fields = somatic_featuremap_fields_transformation.added_format_features

        vcf_reader = pysam.VariantFile(out_file)
        # Check all expected INFO fields are present in header
        for field in tr_info_fields + added_info_fields:
            assert field in vcf_reader.header.info, f"Missing INFO field: {field}"
        for field in expected_format_fields:
            assert field in vcf_reader.header.formats, f"Missing FORMAT field: {field}"

        # Check that records contain the expected fields
        records_with_tr_fields = 0
        records_with_added_fields = 0
        total_records = 0

        for rec in vcf_reader:
            total_records += 1
            # Check if this record has TR fields (from bcftools annotate)
            has_tr_fields = any(field in rec.info for field in tr_info_fields)
            # Check if this record has added fields (from Python processing)
            has_added_fields = any(field in rec.info for field in added_info_fields)

            if has_tr_fields:
                records_with_tr_fields += 1

            if has_added_fields:
                records_with_added_fields += 1
                # Check all added INFO and FORMAT fields are present
                for field in added_info_fields:
                    assert field in rec.info, f"Record missing added INFO field: {field}"
                for field in expected_format_fields:
                    assert field in rec.samples[0], f"Sample[0] missing FORMAT field: {field}"

        # Most records should have TR fields, some should have added fields
        assert records_with_tr_fields > 0, f"No records found with TR fields out of {total_records} records"
        assert records_with_added_fields > 0, f"No records found with added fields out of {total_records} records"

        vcf_reader.close()

    def test_somatic_featuremap_fields_transformation_with_model(self, tmpdir, resources_dir):
        somatic_featuremap_vcf = pjoin(resources_dir, "Pa_46.tumor_normal.merged.tumor_PASS.mpileup_PASS.chr19.vcf.gz")
        ref_tr_file = pjoin(resources_dir, "tr_hg38.chr19.bed")
        interval_list_bed_file = pjoin(resources_dir, "wgs_calling_regions.hg38.chr19_test.interval_list.bed")
        out_dir = tmpdir
        model_file = pjoin(resources_dir, "HG006_HG003.v1.23.5pGenome.t_alt_reads_2-10.json")
        out_file = pjoin(out_dir, "Pa_46.tumor_normal.merged.tumor_PASS.mpileup_PASS.chr19.xgb_proba.vcf.gz")

        # Run the script's main function
        somatic_featuremap_fields_transformation.run(
            [
                "somatic_featuremap_fields_transformation",
                "-sfm",
                somatic_featuremap_vcf,
                "-o",
                out_file,
                "-i",
                interval_list_bed_file,
                "-ref_tr",
                ref_tr_file,
                "-xgb_model",
                model_file,
            ]
        )

        # check that the output file exists and has the expected content
        assert os.path.isfile(out_file)
