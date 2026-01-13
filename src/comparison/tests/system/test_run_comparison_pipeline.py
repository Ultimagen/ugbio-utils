import os
from os.path import dirname
from pathlib import Path

import pytest
from ugbio_comparison import run_comparison_pipeline
from ugbio_core.consts import DEFAULT_FLOW_ORDER
from ugbio_core.h5_utils import read_hdf


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


class TestRunComparisonPipeline:
    def test_run_comparison_pipeline(self, tmpdir, resources_dir):
        output_file = f"{tmpdir}/HG00239.vcf.gz"
        os.makedirs(dirname(output_file), exist_ok=True)
        general_inputs_dir = resources_dir / "chr1_head"
        run_comparison_pipeline.run(
            [
                "run_comparison_pipeline",
                "--input_prefix",
                f"{resources_dir}/004797-UGAv3-51.filtered.chr1_1_1000000",
                "--output_file",
                f"{tmpdir}/004797-UGAv3-51.comp.h5",
                "--output_interval",
                f"{tmpdir}/004797-UGAv3-51.comp.bed",
                "--gtr_vcf",
                f"{resources_dir}/HG004_GRCh38_GIAB_1_22_v4.2.1_benchmark.broad-header.chr1_1_1000000.vcf.gz",
                "--highconf_intervals",
                f"{resources_dir}/HG004_GRCh38_GIAB_1_22_v4.2.1_benchmark_noinconsistent.chr1_1_1000000.bed",
                "--use_tmpdir",
                "--reference",
                f"{general_inputs_dir}/Homo_sapiens_assembly38.fasta",
                "--reference_dict",
                f"{general_inputs_dir}/Homo_sapiens_assembly38.dict",
                "--call_sample_name",
                "UGAv3-51",
                "--truth_sample_name",
                "HG004",
                "--ignore_filter_status",
                "--flow_order",
                DEFAULT_FLOW_ORDER,
                "--annotate_intervals",
                f"{general_inputs_dir}/LCR-hs38.bed",
                "--annotate_intervals",
                f"{general_inputs_dir}/exome.twist.bed",
                "--annotate_intervals",
                f"{general_inputs_dir}/mappability.0.bed",
                "--annotate_intervals",
                f"{general_inputs_dir}/hmers_7_and_higher.bed",
                "--n_jobs",
                "4",
                "--coverage_bw_all_quality",
                f"{resources_dir}/004797-UGAv3-51.chr1.q0.Q0.l0.w1.depth.chr1_1_1000000.bw",
                "--coverage_bw_high_quality",
                f"{resources_dir}/004797-UGAv3-51.chr1.q0.Q20.l0.w1.depth.chr1_1_1000000.bw",
            ]
        )
        output_df = read_hdf(f"{tmpdir}/004797-UGAv3-51.comp.h5", key="chr1")
        assert {"tp": 345, "fn": 30, "fp": 27} == dict(output_df["classify"].value_counts())

    def test_run_comparison_pipeline_sentieon(self, tmpdir, resources_dir):
        output_file = f"{tmpdir}/004777-UGAv3-20.pred.chr1_1_1000000.comp.h5"
        os.makedirs(dirname(output_file), exist_ok=True)
        general_inputs_dir = resources_dir / "chr1_head"

        run_comparison_pipeline.run(
            [
                "run_comparison_pipeline",
                "--input_prefix",
                f"{resources_dir}/004777-UGAv3-20.pred.chr1_1-1000000",
                "--output_file",
                f"{tmpdir}/004777-UGAv3-20.pred.chr1_1_1000000.comp.h5",
                "--output_interval",
                f"{tmpdir}/004777-UGAv3-20.comp.bed",
                "--gtr_vcf",
                f"{resources_dir}/HG001_GRCh38_1_22_v4.2.1_benchmark.chr1_1-1000000.vcf.gz",
                "--highconf_intervals",
                f"{resources_dir}/HG001_GRCh38_1_22_v4.2.1_benchmark.chr1_1-1000000.bed",
                "--reference",
                f"{general_inputs_dir}/Homo_sapiens_assembly38.fasta",
                "--reference_dict",
                f"{general_inputs_dir}/Homo_sapiens_assembly38.dict",
                "--call_sample_name",
                "UGAv3-20",
                "--truth_sample_name",
                "HG001",
                "--flow_order",
                DEFAULT_FLOW_ORDER,
                "--annotate_intervals",
                f"{general_inputs_dir}/LCR-hs38.bed",
                "--annotate_intervals",
                f"{general_inputs_dir}/exome.twist.bed",
                "--annotate_intervals",
                f"{general_inputs_dir}/mappability.0.bed",
                "--annotate_intervals",
                f"{general_inputs_dir}/hmers_7_and_higher.bed",
                "--scoring_field",
                "ML_PROB",
                "--n_jobs",
                "2",
            ]
        )
        output_df = read_hdf(f"{tmpdir}/004777-UGAv3-20.pred.chr1_1_1000000.comp.h5", key="chr1")
        assert {"tp": 305, "fn": 11, "fp": 5} == dict(output_df["classify"].value_counts())

    def test_run_comparison_pipeline_dv(self, tmpdir, resources_dir):
        output_file = f"{tmpdir}/dv.pred.chr1_1-1000000.h5"
        os.makedirs(dirname(output_file), exist_ok=True)
        general_inputs_dir = resources_dir / "chr1_head"

        run_comparison_pipeline.run(
            [
                "run_comparison_pipeline",
                "--input_prefix",
                f"{resources_dir}/dv.pred.chr1_1-1000000",
                "--output_file",
                f"{tmpdir}/dv.pred.chr1_1-1000000.h5",
                "--output_interval",
                f"{tmpdir}/dv.pred.chr1_1-1000000.bed",
                "--gtr_vcf",
                f"{resources_dir}/HG001_GRCh38_1_22_v4.2.1_benchmark.chr1_1-1000000.vcf.gz",
                "--highconf_intervals",
                f"{resources_dir}/HG001_GRCh38_1_22_v4.2.1_benchmark.chr1_1-1000000.bed",
                "--reference",
                f"{general_inputs_dir}/Homo_sapiens_assembly38.fasta",
                "--reference_dict",
                f"{general_inputs_dir}/Homo_sapiens_assembly38.dict",
                "--call_sample_name",
                "sm1",
                "--truth_sample_name",
                "HG001",
                "--flow_order",
                DEFAULT_FLOW_ORDER,
                "--annotate_intervals",
                f"{general_inputs_dir}/LCR-hs38.bed",
                "--annotate_intervals",
                f"{general_inputs_dir}/exome.twist.bed",
                "--annotate_intervals",
                f"{general_inputs_dir}/mappability.0.bed",
                "--annotate_intervals",
                f"{general_inputs_dir}/hmers_7_and_higher.bed",
                "--scoring_field",
                "QUAL",
                "--revert_hom_ref",
                "--ignore_filter_status",
                "--n_jobs",
                "2",
            ]
        )
        output_df = read_hdf(f"{tmpdir}/dv.pred.chr1_1-1000000.h5", key="chr1")
        assert {"tp": 301, "fn": 15, "fp": 790} == dict(output_df["classify"].value_counts())
