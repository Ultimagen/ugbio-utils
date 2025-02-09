from pathlib import Path

import pytest
import ugbio_core.vcfbed.vcftools as vcftools
from ugbio_filtering import filter_variants_pipeline


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


class TestFilterVariantPipeline:
    def test_filter_variants_pipeline(self, tmpdir, resources_dir):
        output_file = f"{tmpdir}/006919_no_frd_chr1_1_5000000_filtered.vcf.gz"
        filter_variants_pipeline.run(
            [
                "--input_file",
                f"{resources_dir}/006919_no_frd_chr1_1_5000000.vcf.gz",
                "--model_file",
                f"{resources_dir}/approximate_gt.model.pkl",
                "--blacklist_cg_insertions",
                "--output_file",
                output_file,
                "--blacklist",
                f"{resources_dir}/blacklist_example.chr1_1_1000000.pkl",
            ]
        )

        result_df = vcftools.get_vcf_df(output_file)
        assert {"LOW_SCORE": 2394, "PASS": 10284} == dict(result_df["filter"].value_counts())

    def test_filter_variants_pipeline_exact_model(self, tmpdir, resources_dir):
        output_file = f"{tmpdir}/036269-NA24143-Z0016.frd_chr1_1_5000000_filtered.vcf.gz"
        filter_variants_pipeline.run(
            [
                "--input_file",
                f"{resources_dir}/036269-NA24143-Z0016.frd_chr1_1_5000000_unfiltered.vcf.gz",
                "--model_file",
                f"{resources_dir}/exact_gt.model.pkl",
                "--output_file",
                output_file,
                "--custom_annotations",
                "LCR",
                "--custom_annotations",
                "MAP_UNIQUE",
                "--custom_annotations",
                "LONG_HMER",
                "--custom_annotations",
                "UG_HCR",
                "--ref_fasta",
                f"{resources_dir}/chr1_head/Homo_sapiens_assembly38.fasta",
                "--treat_multiallelics",
                "--recalibrate_genotype",
            ]
        )

        out_df = vcftools.get_vcf_df(output_file)
        assert {(0, 1): 5210, (0, 0): 5205, (1, 1): 2989, (1, 2): 61, (2, 2): 3} == dict(out_df["gt"].value_counts())
        assert {"PASS": 8289, "LOW_SCORE": 5179} == dict(out_df["filter"].value_counts())

    def test_filter_variants_pipeline_exact_model_no_gt_recal(self, tmpdir, resources_dir):
        output_file = f"{tmpdir}/036269-NA24143-Z0016.frd_chr1_1_5000000_filtered.vcf.gz"
        filter_variants_pipeline.run(
            [
                "--input_file",
                f"{resources_dir}/036269-NA24143-Z0016.frd_chr1_1_5000000_unfiltered.vcf.gz",
                "--model_file",
                f"{resources_dir}/exact_gt.model.pkl",
                "--output_file",
                output_file,
                "--custom_annotations",
                "LCR",
                "--custom_annotations",
                "MAP_UNIQUE",
                "--custom_annotations",
                "LONG_HMER",
                "--custom_annotations",
                "UG_HCR",
                "--ref_fasta",
                f"{resources_dir}/chr1_head/Homo_sapiens_assembly38.fasta",
                "--treat_multiallelics",
            ]
        )

        out_df = vcftools.get_vcf_df(output_file)
        assert {(0, 1): 10086, (1, 1): 3306, (1, 2): 76} == dict(out_df["gt"].value_counts())
        assert {"PASS": 8289, "LOW_SCORE": 5179} == dict(out_df["filter"].value_counts())

    def test_filter_variants_pipeline_blacklist_only(self, tmpdir, resources_dir):
        output_file = f"{tmpdir}/004777-X0024.annotated.AF_chr1_1_1000000_filtered.blacklist_only.vcf.gz"
        filter_variants_pipeline.run(
            [
                "--input_file",
                f"{resources_dir}/004777-X0024.annotated.AF_chr1_1_1000000.vcf.gz",
                "--blacklist_cg_insertions",
                "--output_file",
                output_file,
                "--blacklist",
                f"{resources_dir}/blacklist_example.chr1_1_1000000.pkl",
            ]
        )

        out_df = vcftools.get_vcf_df(output_file)
        assert 4 == out_df[out_df["blacklst"].notna()]["blacklst"].count()

    def test_filter_variants_pipeline_cg_only(self, tmpdir, resources_dir):
        output_file = f"{tmpdir}/004777-X0024.annotated.AF_chr1_1_1000000_filtered.blacklist_only.vcf.gz"
        filter_variants_pipeline.run(
            [
                "--input_file",
                f"{resources_dir}/004777-X0024.annotated.AF_chr1_1_1000000.vcf.gz",
                "--blacklist_cg_insertions",
                "--output_file",
                output_file,
            ]
        )
        out_df = vcftools.get_vcf_df(output_file)
        assert 3 == out_df[out_df["blacklst"].notna()]["blacklst"].count()
