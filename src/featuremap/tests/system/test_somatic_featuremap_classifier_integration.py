"""Integration tests for somatic_featuremap_classifier pipeline.

These tests run the full pipeline with real bioinformatics tools (bcftools, bedtools, tabix).
NO mocking of ugbio_core or external tools - tests validate full integration.
"""

import polars as pl
import pysam
import pytest
from conftest import (
    NORMAL_SAMPLE,
    TUMOR_SAMPLE,
    validate_output_vcf,
)
from ugbio_featuremap.somatic_featuremap_classifier import (
    somatic_featuremap_classifier,
)
from ugbio_featuremap.somatic_featuremap_utils import PILEUP_CONFIG


class TestFullPipeline:
    """End-to-end integration tests for the classifier pipeline."""

    def test_full_pipeline_with_model(self, tmp_path, mini_somatic_vcf, tr_bed, genome_fai, xgb_model_fresh_frozen):
        """Test complete pipeline with V1.15 model on minimal VCF."""
        output_vcf = tmp_path / "output.vcf.gz"
        output_parquet = tmp_path / "output.parquet"

        result_vcf, result_parquet = somatic_featuremap_classifier(
            somatic_featuremap=mini_somatic_vcf,
            output_vcf=output_vcf,
            genome_index_file=genome_fai,
            tandem_repeats_bed=tr_bed,
            xgb_model_json=xgb_model_fresh_frozen,
            output_parquet=output_parquet,
            filter_string="PASS",
            n_threads=1,
        )

        # Validate VCF output
        records = validate_output_vcf(
            result_vcf,
            expected_info_fields=["XGB_PROBA", "TR_DISTANCE"],
            expected_samples=[TUMOR_SAMPLE, NORMAL_SAMPLE],
            min_records=1,
        )

        # Validate XGB_PROBA values are in [0, 1]
        for rec in records:
            xgb_proba = rec.info.get("XGB_PROBA")
            if xgb_proba is not None:
                assert 0.0 <= xgb_proba <= 1.0, f"XGB_PROBA={xgb_proba} out of range at {rec.chrom}:{rec.pos}"

    def test_full_pipeline_with_ffpe_model(self, tmp_path, mini_somatic_vcf, tr_bed, genome_fai, xgb_model_ffpe):
        """Test complete pipeline with FFPE model."""
        output_vcf = tmp_path / "output.vcf.gz"
        output_parquet = tmp_path / "output.parquet"

        result_vcf, result_parquet = somatic_featuremap_classifier(
            somatic_featuremap=mini_somatic_vcf,
            output_vcf=output_vcf,
            genome_index_file=genome_fai,
            tandem_repeats_bed=tr_bed,
            xgb_model_json=xgb_model_ffpe,
            output_parquet=output_parquet,
            filter_string="PASS",
            n_threads=1,
        )

        records = validate_output_vcf(
            result_vcf,
            expected_info_fields=["XGB_PROBA"],
            min_records=1,
        )

        for rec in records:
            xgb_proba = rec.info.get("XGB_PROBA")
            if xgb_proba is not None:
                assert 0.0 <= xgb_proba <= 1.0

    def test_full_pipeline_verbose_saves_debug_parquet(
        self, tmp_path, mini_somatic_vcf, tr_bed, genome_fai, xgb_model_fresh_frozen
    ):
        """Test verbose mode saves full DataFrame with xgb_proba to parquet."""
        output_vcf = tmp_path / "output.vcf.gz"
        output_parquet = tmp_path / "output.parquet"

        somatic_featuremap_classifier(
            somatic_featuremap=mini_somatic_vcf,
            output_vcf=output_vcf,
            genome_index_file=genome_fai,
            tandem_repeats_bed=tr_bed,
            xgb_model_json=xgb_model_fresh_frozen,
            output_parquet=output_parquet,
            filter_string="PASS",
            n_threads=1,
            verbose=True,
        )

        # In verbose mode, parquet should contain xgb_proba
        parquet_df = pl.read_parquet(output_parquet)
        assert "xgb_proba" in parquet_df.columns, "Verbose parquet should contain xgb_proba column"
        assert len(parquet_df) > 0


class TestOutputValidation:
    """Tests validating the structure and content of pipeline outputs."""

    @pytest.fixture(autouse=True)
    def _run_pipeline(self, tmp_path, mini_somatic_vcf, tr_bed, genome_fai, xgb_model_fresh_frozen):
        """Run pipeline once and share outputs across tests in this class."""
        self.output_vcf = tmp_path / "output.vcf.gz"
        self.output_parquet = tmp_path / "output.parquet"

        self.result_vcf, self.result_parquet = somatic_featuremap_classifier(
            somatic_featuremap=mini_somatic_vcf,
            output_vcf=self.output_vcf,
            genome_index_file=genome_fai,
            tandem_repeats_bed=tr_bed,
            xgb_model_json=xgb_model_fresh_frozen,
            output_parquet=self.output_parquet,
            filter_string="PASS",
            n_threads=1,
            verbose=True,
        )

    def test_output_vcf_has_tumor_sample_header(self):
        """Test that output VCF has ##tumor_sample header line."""
        with pysam.VariantFile(str(self.result_vcf)) as vcf:
            header_str = str(vcf.header)
            assert f"##tumor_sample={TUMOR_SAMPLE}" in header_str

    def test_output_vcf_has_tr_fields(self):
        """Test TR INFO fields are present in output VCF header."""
        validate_output_vcf(
            self.result_vcf,
            expected_info_fields=["TR_START", "TR_END", "TR_SEQ", "TR_LENGTH", "TR_SEQ_UNIT_LENGTH", "TR_DISTANCE"],
        )

    def test_output_vcf_records_have_xgb_proba(self):
        """Every record in the output should have XGB_PROBA."""
        with pysam.VariantFile(str(self.result_vcf)) as vcf:
            for rec in vcf:
                assert "XGB_PROBA" in rec.info, f"Missing XGB_PROBA at {rec.chrom}:{rec.pos}"

    def test_output_vcf_only_pass_variants(self):
        """When filter_string='PASS', output should only contain PASS variants."""
        with pysam.VariantFile(str(self.result_vcf)) as vcf:
            for rec in vcf:
                assert (
                    "PASS" in rec.filter
                ), f"Non-PASS variant in output: {rec.chrom}:{rec.pos} filter={list(rec.filter)}"

    def test_output_parquet_has_required_columns(self):
        """Parquet output should have core variant columns and xgb_proba."""
        parquet_df = pl.read_parquet(self.result_parquet)

        core_columns = {"CHROM", "POS", "REF", "ALT"}
        assert core_columns.issubset(
            set(parquet_df.columns)
        ), f"Missing core columns: {core_columns - set(parquet_df.columns)}"
        assert "xgb_proba" in parquet_df.columns

    def test_output_parquet_has_aggregated_features(self):
        """Parquet should contain aggregated features for both samples."""
        parquet_df = pl.read_parquet(self.result_parquet)

        tumor_suffix = f"_{TUMOR_SAMPLE}"
        normal_suffix = f"_{NORMAL_SAMPLE}"

        # Check some key aggregated columns for tumor
        for prefix in ["MQUAL_mean", "SNVQ_mean", "MAPQ_mean", "EDIST_mean", "RL_mean"]:
            assert f"{prefix}{tumor_suffix}" in parquet_df.columns, f"Missing {prefix}{tumor_suffix}"
            assert f"{prefix}{normal_suffix}" in parquet_df.columns, f"Missing {prefix}{normal_suffix}"

    def test_output_parquet_has_derived_columns(self):
        """Parquet should contain derived columns (duplicate counts, strand counts, etc.)."""
        parquet_df = pl.read_parquet(self.result_parquet)

        for sample in [TUMOR_SAMPLE, NORMAL_SAMPLE]:
            s = f"_{sample}"
            for col in [
                f"DUP_count_non_zero{s}",
                f"DUP_count_zero{s}",
                f"REV_count_non_zero{s}",
                f"REV_count_zero{s}",
                f"FILT_count_non_zero{s}",
                f"SCST_count_non_zero{s}",
                f"SCED_count_non_zero{s}",
            ]:
                assert col in parquet_df.columns, f"Missing derived column: {col}"

    def test_output_parquet_has_pileup_features(self):
        """Parquet should contain REF_{pos} and NON_REF_{pos} for both samples."""
        parquet_df = pl.read_parquet(self.result_parquet)

        for sample in [TUMOR_SAMPLE, NORMAL_SAMPLE]:
            s = f"_{sample}"
            for pos in PILEUP_CONFIG.positions:
                assert f"REF_{pos}{s}" in parquet_df.columns, f"Missing REF_{pos}{s}"
                assert f"NON_REF_{pos}{s}" in parquet_df.columns, f"Missing NON_REF_{pos}{s}"


class TestFilterVariations:
    """Test different filter_string options."""

    def test_no_filter(self, tmp_path, mini_somatic_vcf, tr_bed, genome_fai, xgb_model_fresh_frozen):
        """Test pipeline without filter (all variants processed)."""
        output_vcf = tmp_path / "output.vcf.gz"

        result_vcf, _ = somatic_featuremap_classifier(
            somatic_featuremap=mini_somatic_vcf,
            output_vcf=output_vcf,
            genome_index_file=genome_fai,
            tandem_repeats_bed=tr_bed,
            xgb_model_json=xgb_model_fresh_frozen,
            filter_string="",
            n_threads=1,
        )

        # Without filter, should include PASS + PreFiltered + SingleRead
        records = validate_output_vcf(result_vcf, min_records=1)
        # Should have more variants than just PASS
        with pysam.VariantFile(str(mini_somatic_vcf)) as vcf:
            total_in_input = sum(1 for _ in vcf)
        assert (
            len(records) == total_in_input
        ), f"Without filter, expected all {total_in_input} variants, got {len(records)}"
