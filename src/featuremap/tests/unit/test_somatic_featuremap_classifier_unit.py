"""Unit tests for somatic_featuremap_classifier transformation logic.

These tests validate individual pipeline steps with real tools (no mocking of ugbio_core).
Each step is tested independently using minimal test data.
"""

import polars as pl
import pysam
import pytest
from conftest import NORMAL_SAMPLE, TUMOR_SAMPLE
from ugbio_featuremap.featuremap_utils import FeatureMapFields, TandemRepeatFields
from ugbio_featuremap.somatic_featuremap_classifier import (
    _get_xgb_proba_bcftools_columns,
    aggregated_df_post_processing,
    calculate_ref_nonref_columns,
    filter_and_annotate_tr,
    get_fields_to_drop_from_vcf,
    read_vcf_with_aggregation,
    rename_cols_for_model,
    run_classifier,
    somatic_featuremap_classifier,
)
from ugbio_featuremap.somatic_featuremap_utils import (
    PILEUP_CONFIG,
    REQUIRED_FORMAT_FIELDS,
    REQUIRED_INFO_FIELDS,
)


class TestFilterAndAnnotateTR:
    """Test Step 1: Filter + TR annotation."""

    def test_filter_pass_only(self, tmp_path, mini_somatic_vcf, tr_bed, genome_fai):
        """Filter VCF to PASS variants and annotate with TR info."""
        result_vcf = filter_and_annotate_tr(
            input_vcf=mini_somatic_vcf,
            tandem_repeats_bed=tr_bed,
            genome_index_file=genome_fai,
            out_dir=tmp_path,
            filter_string="PASS",
            n_threads=1,
        )

        assert result_vcf.exists()
        assert ".filtered.tr_info.vcf.gz" in result_vcf.name

        # All records should be PASS
        with pysam.VariantFile(str(result_vcf)) as vcf:
            for rec in vcf:
                assert "PASS" in rec.filter

            # Check TR fields are in header
            assert "TR_DISTANCE" in vcf.header.info

    def test_filter_no_filter(self, tmp_path, mini_somatic_vcf, tr_bed, genome_fai):
        """TR annotation without filtering (all variants kept)."""
        result_vcf = filter_and_annotate_tr(
            input_vcf=mini_somatic_vcf,
            tandem_repeats_bed=tr_bed,
            genome_index_file=genome_fai,
            out_dir=tmp_path,
            filter_string=None,
            n_threads=1,
        )

        assert result_vcf.exists()
        assert ".tr_info.vcf.gz" in result_vcf.name

        # Should have all variants (including PreFiltered and SingleRead)
        with pysam.VariantFile(str(result_vcf)) as vcf:
            records = list(vcf)
        assert len(records) >= 20, f"Expected all variants, got {len(records)}"

    def test_tr_annotation_fields_populated(self, tmp_path, mini_somatic_vcf, tr_bed, genome_fai):
        """Verify TR INFO fields are populated in at least some records."""
        result_vcf = filter_and_annotate_tr(
            input_vcf=mini_somatic_vcf,
            tandem_repeats_bed=tr_bed,
            genome_index_file=genome_fai,
            out_dir=tmp_path,
            filter_string="PASS",
            n_threads=1,
        )

        tr_fields = [f.value for f in TandemRepeatFields]
        records_with_tr = 0
        total_records = 0
        with pysam.VariantFile(str(result_vcf)) as vcf:
            # All TR fields should be declared in header
            for field in tr_fields:
                assert field in vcf.header.info, f"TR field {field} missing from header"

            for rec in vcf:
                total_records += 1
                if any(field in rec.info for field in tr_fields):
                    records_with_tr += 1

        # TR_DISTANCE should be populated for every variant (distance to nearest TR)
        assert records_with_tr > 0, f"No records with TR fields out of {total_records}"


class TestGetColumnsToDropFromVcf:
    """Test column selection logic for VCF → DataFrame conversion."""

    def test_required_fields_are_kept(self, mini_somatic_vcf):
        """Fields in REQUIRED_INFO_FIELDS and REQUIRED_FORMAT_FIELDS should not be dropped."""
        drop_info, drop_format = get_fields_to_drop_from_vcf(mini_somatic_vcf)

        # None of the required fields should be in the drop sets
        assert not (
            REQUIRED_INFO_FIELDS & drop_info
        ), f"Required INFO fields would be dropped: {REQUIRED_INFO_FIELDS & drop_info}"
        assert not (
            REQUIRED_FORMAT_FIELDS & drop_format
        ), f"Required FORMAT fields would be dropped: {REQUIRED_FORMAT_FIELDS & drop_format}"

    def test_non_required_fields_are_dropped(self, mini_somatic_vcf):
        """Fields not in REQUIRED sets should be in drop sets."""
        drop_info, drop_format = get_fields_to_drop_from_vcf(mini_somatic_vcf)

        # Non-required fields like X_HMER_REF, RPA, etc. should be dropped
        assert len(drop_info) > 0, "Expected some INFO fields to be dropped"
        assert len(drop_format) > 0, "Expected some FORMAT fields to be dropped"


class TestReadVcfWithAggregation:
    """Test Step 2: VCF → DataFrame conversion with aggregation."""

    @pytest.fixture
    def filtered_vcf(self, tmp_path, mini_somatic_vcf, tr_bed, genome_fai):
        """Pre-filtered VCF with TR annotations for aggregation tests."""
        return filter_and_annotate_tr(
            input_vcf=mini_somatic_vcf,
            tandem_repeats_bed=tr_bed,
            genome_index_file=genome_fai,
            out_dir=tmp_path,
            filter_string="PASS",
            n_threads=1,
        )

    def test_read_vcf_produces_dataframe(self, tmp_path, filtered_vcf):
        """Test that VCF to DataFrame conversion produces non-empty result."""
        output_parquet = tmp_path / "test.parquet"

        variants_df = read_vcf_with_aggregation(
            vcf_path=filtered_vcf,
            output_parquet_path=output_parquet,
            tumor_sample=TUMOR_SAMPLE,
            normal_sample=NORMAL_SAMPLE,
            n_threads=1,
        )

        assert isinstance(variants_df, pl.DataFrame)
        assert len(variants_df) > 0
        assert output_parquet.exists()

    def test_core_variant_columns_present(self, tmp_path, filtered_vcf):
        """DataFrame should have core variant identification columns."""
        output_parquet = tmp_path / "test.parquet"

        variants_df = read_vcf_with_aggregation(
            vcf_path=filtered_vcf,
            output_parquet_path=output_parquet,
            tumor_sample=TUMOR_SAMPLE,
            normal_sample=NORMAL_SAMPLE,
            n_threads=1,
        )

        for col in ["CHROM", "POS", "REF", "ALT"]:
            assert col in variants_df.columns, f"Missing core column: {col}"

    def test_aggregated_columns_present(self, tmp_path, filtered_vcf):
        """DataFrame should have aggregated FORMAT columns for both samples."""
        output_parquet = tmp_path / "test.parquet"

        variants_df = read_vcf_with_aggregation(
            vcf_path=filtered_vcf,
            output_parquet_path=output_parquet,
            tumor_sample=TUMOR_SAMPLE,
            normal_sample=NORMAL_SAMPLE,
            n_threads=1,
        )

        # Check aggregation columns for list-type fields
        for field in ["MQUAL", "SNVQ", "MAPQ", "EDIST", "RL"]:
            for agg in ["mean", "min", "max"]:
                col = f"{field}_{agg}_{TUMOR_SAMPLE}"
                assert col in variants_df.columns, f"Missing aggregated column: {col}"

    def test_pileup_ref_nonref_columns_present(self, tmp_path, filtered_vcf):
        """DataFrame should have REF_{pos} and NON_REF_{pos} columns for both samples."""
        output_parquet = tmp_path / "test.parquet"

        variants_df = read_vcf_with_aggregation(
            vcf_path=filtered_vcf,
            output_parquet_path=output_parquet,
            tumor_sample=TUMOR_SAMPLE,
            normal_sample=NORMAL_SAMPLE,
            n_threads=1,
        )

        for sample in [TUMOR_SAMPLE, NORMAL_SAMPLE]:
            for pos in PILEUP_CONFIG.positions:
                assert f"REF_{pos}_{sample}" in variants_df.columns, f"Missing REF_{pos}_{sample}"
                assert f"NON_REF_{pos}_{sample}" in variants_df.columns, f"Missing NON_REF_{pos}_{sample}"

    def test_derived_columns_present(self, tmp_path, filtered_vcf):
        """DataFrame should have derived columns from post-processing."""
        output_parquet = tmp_path / "test.parquet"

        variants_df = read_vcf_with_aggregation(
            vcf_path=filtered_vcf,
            output_parquet_path=output_parquet,
            tumor_sample=TUMOR_SAMPLE,
            normal_sample=NORMAL_SAMPLE,
            n_threads=1,
        )

        for sample in [TUMOR_SAMPLE, NORMAL_SAMPLE]:
            s = f"_{sample}"
            for col_base in [
                "DUP_count_non_zero",
                "DUP_count_zero",
                "REV_count_non_zero",
                "REV_count_zero",
                "FILT_count_non_zero",
                "SCST_count_non_zero",
                "SCED_count_non_zero",
            ]:
                assert f"{col_base}{s}" in variants_df.columns, f"Missing derived column: {col_base}{s}"


class TestCalculateRefNonrefColumns:
    """Test PILEUP ref/nonref calculations."""

    @pytest.fixture
    def sample_pileup_df(self):
        """Create a minimal DataFrame with PILEUP columns for testing."""
        suffix = f"_{TUMOR_SAMPLE}"
        data = {
            "CHROM": ["chr19"],
            "POS": [1000000],
            "REF": ["A"],
            "ALT": ["T"],
            FeatureMapFields.X_PREV2.value: ["C"],
            FeatureMapFields.X_PREV1.value: ["G"],
            FeatureMapFields.X_NEXT1.value: ["T"],
            FeatureMapFields.X_NEXT2.value: ["A"],
        }

        # Add PILEUP columns for all bases and positions
        for pos in PILEUP_CONFIG.positions:
            for base in PILEUP_CONFIG.bases:
                col_name = f"PILEUP_{base}_{pos}{suffix}"
                data[col_name] = [10]  # Default count of 10 for each
            for indel in PILEUP_CONFIG.indels:
                col_name = f"PILEUP_{indel}_{pos}{suffix}"
                data[col_name] = [1]  # Small indel count

        return pl.DataFrame(data)

    def test_ref_nonref_columns_created(self, sample_pileup_df):
        """Test that REF_{pos} and NON_REF_{pos} columns are created."""
        suffix = f"_{TUMOR_SAMPLE}"
        result = calculate_ref_nonref_columns(sample_pileup_df, suffix)

        for pos in PILEUP_CONFIG.positions:
            assert f"REF_{pos}{suffix}" in result.columns
            assert f"NON_REF_{pos}{suffix}" in result.columns

    def test_ref_column_matches_reference_base(self, sample_pileup_df):
        """REF column should equal the PILEUP count for the reference base at that position."""
        suffix = f"_{TUMOR_SAMPLE}"
        result = calculate_ref_nonref_columns(sample_pileup_df, suffix)

        # Position C: REF=A, so REF_C should be PILEUP_A_C
        ref_c_value = result[f"REF_C{suffix}"][0]
        pileup_a_c = result[f"PILEUP_A_C{suffix}"][0]
        assert ref_c_value == pileup_a_c, f"REF_C ({ref_c_value}) != PILEUP_A_C ({pileup_a_c})"

    def test_nonref_is_sum_of_nonref_bases_plus_indels(self, sample_pileup_df):
        """NON_REF should be sum of non-reference bases + DEL + INS."""
        suffix = f"_{TUMOR_SAMPLE}"
        result = calculate_ref_nonref_columns(sample_pileup_df, suffix)

        # Position C: REF=A
        # NON_REF_C = PILEUP_C_C + PILEUP_G_C + PILEUP_T_C + PILEUP_DEL_C + PILEUP_INS_C
        expected_nonref = 10 + 10 + 10 + 1 + 1  # C, G, T (not A) + DEL + INS = 32
        actual_nonref = result[f"NON_REF_C{suffix}"][0]
        assert actual_nonref == expected_nonref, f"NON_REF_C ({actual_nonref}) != expected ({expected_nonref})"

    def test_ref_nonref_sum_invariant(self, sample_pileup_df):
        """REF + NON_REF should equal sum of all PILEUP columns for that position."""
        suffix = f"_{TUMOR_SAMPLE}"
        result = calculate_ref_nonref_columns(sample_pileup_df, suffix)

        for pos in PILEUP_CONFIG.positions:
            ref_val = result[f"REF_{pos}{suffix}"][0]
            nonref_val = result[f"NON_REF_{pos}{suffix}"][0]
            total_pileup = sum(
                result[PILEUP_CONFIG.get_column_name(elem, pos, suffix)][0]
                for elem in list(PILEUP_CONFIG.bases) + list(PILEUP_CONFIG.indels)
            )
            assert (
                ref_val + nonref_val == total_pileup
            ), f"Position {pos}: REF({ref_val}) + NON_REF({nonref_val}) != total({total_pileup})"


class TestAggregatedDfPostProcessing:
    """Test derived column calculations in post-processing."""

    @pytest.fixture
    def sample_agg_df(self):
        """Create a minimal aggregated DataFrame for post-processing tests."""
        data = {}
        for sample in [TUMOR_SAMPLE, NORMAL_SAMPLE]:
            s = f"_{sample}"
            data[f"DUP_count{s}"] = [10]
            data[f"DUP_count_zero{s}"] = [4]
            data[f"REV_count{s}"] = [10]
            data[f"REV_count_zero{s}"] = [6]
            data[f"FILT_count{s}"] = [10]
            data[f"FILT_count_zero{s}"] = [2]
            data[f"SCST_count{s}"] = [10]
            data[f"SCST_count_zero{s}"] = [7]
            data[f"SCED_count{s}"] = [10]
            data[f"SCED_count_zero{s}"] = [8]

        return pl.DataFrame(data)

    def test_count_non_zero_calculation(self, sample_agg_df):
        """All COUNT_NON_ZERO columns = count - count_zero."""
        result = aggregated_df_post_processing(sample_agg_df, [TUMOR_SAMPLE, NORMAL_SAMPLE])

        for sample in [TUMOR_SAMPLE, NORMAL_SAMPLE]:
            s = f"_{sample}"
            assert result[f"DUP_count_non_zero{s}"][0] == 10 - 4
            assert result[f"REV_count_non_zero{s}"][0] == 10 - 6
            assert result[f"FILT_count_non_zero{s}"][0] == 10 - 2
            assert result[f"SCST_count_non_zero{s}"][0] == 10 - 7
            assert result[f"SCED_count_non_zero{s}"][0] == 10 - 8


class TestRenameColsForModel:
    """Test Step 3: Column renaming for model inference."""

    @pytest.fixture
    def sample_df_for_rename(self):
        """Create a DataFrame with sample-suffixed columns ready for renaming."""
        data = {
            "CHROM": ["chr19"],
            "POS": [1000000],
            "REF": ["A"],
            "ALT": ["T"],
            "TR_DISTANCE": [100],
        }

        for sample, prefix in [(TUMOR_SAMPLE, "t_"), (NORMAL_SAMPLE, "n_")]:
            s = f"_{sample}"
            # Aggregation columns
            for field in ["MQUAL", "SNVQ", "MAPQ", "EDIST", "RL"]:
                data[f"{field}_mean{s}"] = [30.0]
                data[f"{field}_min{s}"] = [20.0]
                data[f"{field}_max{s}"] = [40.0]

            data[f"MAPQ_count_zero{s}"] = [2]
            data[f"AD_1{s}"] = [5]
            data[f"DP{s}"] = [100]
            data[f"VAF{s}"] = [0.05]
            data[f"RAW_VAF{s}"] = [0.04]
            data[f"DP_FILT{s}"] = [95]

            # Derived columns
            data[f"DUP_count_non_zero{s}"] = [3]
            data[f"DUP_count_zero{s}"] = [7]
            data[f"REV_count_non_zero{s}"] = [4]
            data[f"REV_count_zero{s}"] = [6]
            data[f"FILT_count_non_zero{s}"] = [8]
            data[f"SCST_count_non_zero{s}"] = [2]
            data[f"SCED_count_non_zero{s}"] = [1]

            # ref/nonref columns
            for pos in PILEUP_CONFIG.positions:
                data[f"REF_{pos}{s}"] = [50]
                data[f"NON_REF_{pos}{s}"] = [5]

        return pl.DataFrame(data)

    def test_rename_produces_t_n_prefixed_columns(self, sample_df_for_rename):
        """Renamed columns should use t_/n_ prefix convention."""
        result = rename_cols_for_model(sample_df_for_rename, [TUMOR_SAMPLE, NORMAL_SAMPLE])

        # Tumor columns
        assert "t_mqual_mean" in result.columns
        assert "t_snvq_mean" in result.columns
        assert "t_alt_reads" in result.columns
        assert "t_dp" in result.columns
        assert "t_vaf" in result.columns
        assert "t_ref0" in result.columns
        assert "t_nonref0" in result.columns

        # Normal columns
        assert "n_mqual_mean" in result.columns
        assert "n_alt_reads" in result.columns
        assert "n_dp" in result.columns
        assert "n_vaf" in result.columns

    def test_rename_preserves_data_values(self, sample_df_for_rename):
        """Data values should be preserved after renaming."""
        result = rename_cols_for_model(sample_df_for_rename, [TUMOR_SAMPLE, NORMAL_SAMPLE])

        assert result["t_mqual_mean"][0] == 30.0
        assert result["t_alt_reads"][0] == 5
        assert result["t_dp"][0] == 100
        assert result["t_vaf"][0] == 0.05

    def test_rename_adds_ref_alt_allele_columns(self, sample_df_for_rename):
        """Renamed DataFrame should have ref_allele and alt_allele columns."""
        result = rename_cols_for_model(sample_df_for_rename, [TUMOR_SAMPLE, NORMAL_SAMPLE])

        assert "ref_allele" in result.columns
        assert "alt_allele" in result.columns
        assert result["ref_allele"][0] == "A"
        assert result["alt_allele"][0] == "T"

    def test_rename_adds_tr_distance(self, sample_df_for_rename):
        """t_tr_distance should be added from TR_DISTANCE INFO field."""
        result = rename_cols_for_model(sample_df_for_rename, [TUMOR_SAMPLE, NORMAL_SAMPLE])

        assert "t_tr_distance" in result.columns
        assert result["t_tr_distance"][0] == 100

    def test_rename_missing_columns_handled_gracefully(self):
        """Missing columns should not cause errors (only existing columns are renamed)."""
        minimal_df = pl.DataFrame(
            {
                "CHROM": ["chr19"],
                "POS": [1000000],
                "REF": ["A"],
                "ALT": ["T"],
            }
        )

        # Should not raise, even though most expected columns are missing
        result = rename_cols_for_model(minimal_df, [TUMOR_SAMPLE, NORMAL_SAMPLE])
        assert "ref_allele" in result.columns


class TestRunClassifier:
    """Test Step 3: XGBoost classifier execution."""

    def test_run_classifier_returns_predictions(
        self, tmp_path, mini_somatic_vcf, tr_bed, genome_fai, xgb_model_fresh_frozen
    ):
        """Classifier should return a Series with probabilities in [0, 1]."""
        # First run the pipeline steps to get a proper DataFrame
        filtered_vcf = filter_and_annotate_tr(
            input_vcf=mini_somatic_vcf,
            tandem_repeats_bed=tr_bed,
            genome_index_file=genome_fai,
            out_dir=tmp_path,
            filter_string="PASS",
            n_threads=1,
        )

        output_parquet = tmp_path / "test.parquet"
        aggregated_df = read_vcf_with_aggregation(
            vcf_path=filtered_vcf,
            output_parquet_path=output_parquet,
            tumor_sample=TUMOR_SAMPLE,
            normal_sample=NORMAL_SAMPLE,
            n_threads=1,
        )

        renamed_df = rename_cols_for_model(aggregated_df, [TUMOR_SAMPLE, NORMAL_SAMPLE])
        predictions = run_classifier(renamed_df, xgb_model_fresh_frozen)

        assert isinstance(predictions, pl.Series)
        assert len(predictions) == len(aggregated_df)
        assert predictions.name == "xgb_proba"

        # All predictions should be in [0, 1]
        assert (predictions >= 0.0).all()
        assert (predictions <= 1.0).all()

    def test_model_feature_compatibility(self, tmp_path, mini_somatic_vcf, tr_bed, genome_fai, xgb_model_fresh_frozen):
        """DataFrame columns after renaming should include all model features."""
        from ugbio_featuremap import somatic_featuremap_inference_utils

        # Run pipeline steps
        filtered_vcf = filter_and_annotate_tr(
            input_vcf=mini_somatic_vcf,
            tandem_repeats_bed=tr_bed,
            genome_index_file=genome_fai,
            out_dir=tmp_path,
            filter_string="PASS",
            n_threads=1,
        )

        output_parquet = tmp_path / "test.parquet"
        aggregated_df = read_vcf_with_aggregation(
            vcf_path=filtered_vcf,
            output_parquet_path=output_parquet,
            tumor_sample=TUMOR_SAMPLE,
            normal_sample=NORMAL_SAMPLE,
            n_threads=1,
        )

        renamed_df = rename_cols_for_model(aggregated_df, [TUMOR_SAMPLE, NORMAL_SAMPLE])

        # Load model and check features
        xgb_clf = somatic_featuremap_inference_utils.load_xgb_model(xgb_model_fresh_frozen)
        model_features = set(xgb_clf.get_booster().feature_names)
        df_features = set(renamed_df.columns)

        missing = model_features - df_features
        assert len(missing) == 0, f"DataFrame missing model features: {sorted(missing)}"


class TestXgbProbaBcftoolsColumns:
    """Test VCF annotation helper functions."""

    def test_get_xgb_proba_bcftools_columns_format(self):
        """Column string should follow bcftools annotate format."""
        columns = _get_xgb_proba_bcftools_columns()

        assert columns == "CHROM,POS,REF,ALT,INFO/XGB_PROBA"


class TestFilterStringValidation:
    """Test filter_string validation in somatic_featuremap_classifier."""

    def test_invalid_filter_string_with_space_raises(
        self, tmp_path, mini_somatic_vcf, tr_bed, genome_fai, xgb_model_fresh_frozen
    ):
        """filter_string with spaces raises ValueError."""
        output_vcf = tmp_path / "output.vcf.gz"
        with pytest.raises(ValueError, match="filter_string must contain only"):
            somatic_featuremap_classifier(
                somatic_featuremap=mini_somatic_vcf,
                output_vcf=output_vcf,
                genome_index_file=genome_fai,
                tandem_repeats_bed=tr_bed,
                xgb_model_json=xgb_model_fresh_frozen,
                filter_string="PASS ",
            )

    def test_invalid_filter_string_with_special_chars_raises(
        self, tmp_path, mini_somatic_vcf, tr_bed, genome_fai, xgb_model_fresh_frozen
    ):
        """filter_string with non-standard characters raises ValueError."""
        output_vcf = tmp_path / "output.vcf.gz"
        with pytest.raises(ValueError, match="filter_string must contain only"):
            somatic_featuremap_classifier(
                somatic_featuremap=mini_somatic_vcf,
                output_vcf=output_vcf,
                genome_index_file=genome_fai,
                tandem_repeats_bed=tr_bed,
                xgb_model_json=xgb_model_fresh_frozen,
                filter_string="foo;bar",
            )


class TestInputFileValidation:
    """Test input file existence validation in initialization."""

    def test_missing_somatic_featuremap_raises(self, tmp_path, tr_bed, genome_fai, xgb_model_fresh_frozen):
        """Missing somatic_featuremap raises FileNotFoundError."""
        output_vcf = tmp_path / "output.vcf.gz"
        missing_vcf = tmp_path / "nonexistent.vcf.gz"
        with pytest.raises(FileNotFoundError, match="Input file does not exist"):
            somatic_featuremap_classifier(
                somatic_featuremap=missing_vcf,
                output_vcf=output_vcf,
                genome_index_file=genome_fai,
                tandem_repeats_bed=tr_bed,
                xgb_model_json=xgb_model_fresh_frozen,
            )

    def test_missing_genome_index_raises(self, tmp_path, mini_somatic_vcf, tr_bed, xgb_model_fresh_frozen):
        """Missing genome_index_file raises FileNotFoundError."""
        output_vcf = tmp_path / "output.vcf.gz"
        missing_fai = tmp_path / "nonexistent.fai"
        with pytest.raises(FileNotFoundError, match="Input file does not exist"):
            somatic_featuremap_classifier(
                somatic_featuremap=mini_somatic_vcf,
                output_vcf=output_vcf,
                genome_index_file=missing_fai,
                tandem_repeats_bed=tr_bed,
                xgb_model_json=xgb_model_fresh_frozen,
            )

    def test_missing_tandem_repeats_bed_raises(self, tmp_path, mini_somatic_vcf, genome_fai, xgb_model_fresh_frozen):
        """Missing tandem_repeats_bed raises FileNotFoundError."""
        output_vcf = tmp_path / "output.vcf.gz"
        missing_bed = tmp_path / "nonexistent.bed"
        with pytest.raises(FileNotFoundError, match="Input file does not exist"):
            somatic_featuremap_classifier(
                somatic_featuremap=mini_somatic_vcf,
                output_vcf=output_vcf,
                genome_index_file=genome_fai,
                tandem_repeats_bed=missing_bed,
                xgb_model_json=xgb_model_fresh_frozen,
            )

    def test_missing_xgb_model_raises(self, tmp_path, mini_somatic_vcf, tr_bed, genome_fai):
        """Missing xgb_model_json raises FileNotFoundError."""
        output_vcf = tmp_path / "output.vcf.gz"
        missing_model = tmp_path / "nonexistent.json"
        with pytest.raises(FileNotFoundError, match="Input file does not exist"):
            somatic_featuremap_classifier(
                somatic_featuremap=mini_somatic_vcf,
                output_vcf=output_vcf,
                genome_index_file=genome_fai,
                tandem_repeats_bed=tr_bed,
                xgb_model_json=missing_model,
            )

    def test_missing_regions_bed_raises(self, tmp_path, mini_somatic_vcf, tr_bed, genome_fai, xgb_model_fresh_frozen):
        """Missing regions_bed_file raises FileNotFoundError when provided."""
        output_vcf = tmp_path / "output.vcf.gz"
        missing_bed = tmp_path / "nonexistent.bed"
        with pytest.raises(FileNotFoundError, match="Input file does not exist"):
            somatic_featuremap_classifier(
                somatic_featuremap=mini_somatic_vcf,
                output_vcf=output_vcf,
                genome_index_file=genome_fai,
                tandem_repeats_bed=tr_bed,
                xgb_model_json=xgb_model_fresh_frozen,
                regions_bed_file=missing_bed,
            )
