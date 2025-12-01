from os.path import join as pjoin
from pathlib import Path

import pandas as pd
import pytest
from ugbio_cnv import convert_combined_cnv_results_to_output_formats
from ugbio_cnv.convert_combined_cnv_results_to_output_formats import process_filter_columns
from ugbio_core.test_utils import compare_vcfs


@pytest.fixture
def resources_dir():
    return Path(__file__).parent.parent / "resources"


def compare_strings_ignoring_order(str1, str2, delimiter=","):
    set1 = sorted(str1.split(delimiter))
    set2 = sorted(str2.split(delimiter))
    return set1 == set2


class TestConvertCombinedCnvResultsToOutputFormats:
    test_filter_columns_registry = ["jalign_filter", "LCR_label_value"]
    test_filter_tag_registry = {
        "Clusters": ("Clusters", None, None, "Overlaps with locations with frequent clusters of CNV", "FILTER"),
        "Coverage-Mappability": (
            "Coverage-Mappability",
            None,
            None,
            "Overlaps with low coverage or low mappability regions",
            "FILTER",
        ),
        "Telomere_Centromere": (
            "Telomere_Centromere",
            None,
            None,
            "Overlaps with telomere or centromere regions",
            "FILTER",
        ),
        "NO_JUMP_ALIGNMENT": ("NO_JUMP_ALIGNMENT", None, None, "No jump alignment support", "FILTER"),
        "Filter1": ("Filter1", None, None, "Test Filter 1", "FILTER"),
        "Filter2": ("Filter2", None, None, "Test Filter 2", "FILTER"),
        "Filter3": ("Filter3", None, None, "Test Filter 3", "FILTER"),
    }

    def test_single_filter_value(self):
        """Test processing a single filter value."""
        row = pd.Series({"jalign_filter": "NO_JUMP_ALIGNMENT", "LCR_label_value": "."})
        result = process_filter_columns(
            row,
            filter_columns_registry=self.test_filter_columns_registry,
            filter_tags_registry=self.test_filter_tag_registry,
        )
        assert compare_strings_ignoring_order(result, "NO_JUMP_ALIGNMENT")

    def test_multiple_filters_pipe_separator(self):
        """Test processing multiple filter values separated by |."""
        row = pd.Series({"jalign_filter": "Filter1|Filter2", "LCR_label_value": "PASS"})
        result = process_filter_columns(
            row,
            filter_columns_registry=self.test_filter_columns_registry,
            filter_tags_registry=self.test_filter_tag_registry,
        )
        assert compare_strings_ignoring_order(result, "Filter1,Filter2")

    def test_multiple_filters_comma_separator(self):
        """Test processing multiple filter values separated by ,."""
        row = pd.Series({"jalign_filter": "Filter1;Filter2", "LCR_label_value": "."})
        result = process_filter_columns(
            row,
            filter_columns_registry=self.test_filter_columns_registry,
            filter_tags_registry=self.test_filter_tag_registry,
        )
        assert compare_strings_ignoring_order(result, "Filter1,Filter2")

    def test_mixed_separators(self):
        """Test processing values with mixed | and , separators."""
        row = pd.Series({"jalign_filter": "Filter1|Filter2;Filter3", "LCR_label_value": "PASS"})
        result = process_filter_columns(
            row,
            filter_columns_registry=self.test_filter_columns_registry,
            filter_tags_registry=self.test_filter_tag_registry,
        )
        assert compare_strings_ignoring_order(result, "Filter1,Filter2,Filter3")

    def test_duplicate_filter_removal(self):
        """Test that duplicate filters are removed."""
        row = pd.Series({"jalign_filter": "Filter1|Filter1", "LCR_label_value": "Filter1"})
        result = process_filter_columns(
            row,
            filter_columns_registry=self.test_filter_columns_registry,
            filter_tags_registry=self.test_filter_tag_registry,
        )
        assert result == "Filter1"

    def test_excluded_values_filtered_out(self):
        """Test that PASS, ., and nan values are filtered out."""
        row = pd.Series({"jalign_filter": "Filter1|PASS|.", "LCR_label_value": "PASS"})
        result = process_filter_columns(
            row,
            filter_columns_registry=self.test_filter_columns_registry,
            filter_tags_registry=self.test_filter_tag_registry,
        )
        assert result == "Filter1"

    def test_all_excluded_values_returns_pass(self):
        """Test that when all values are excluded, PASS is returned."""
        row = pd.Series({"jalign_filter": "PASS", "LCR_label_value": "."})
        result = process_filter_columns(
            row,
            filter_columns_registry=self.test_filter_columns_registry,
            filter_tags_registry=self.test_filter_tag_registry,
        )
        assert result == "PASS"

    def test_empty_values_returns_pass(self):
        """Test that empty/null values return PASS."""
        row = pd.Series({"jalign_filter": None, "LCR_label_value": pd.NA})
        result = process_filter_columns(
            row,
            filter_columns_registry=self.test_filter_columns_registry,
            filter_tags_registry=self.test_filter_tag_registry,
        )
        assert result == "PASS"

    def test_filters_from_multiple_columns(self):
        """Test combining filters from multiple columns."""
        row = pd.Series({"jalign_filter": "Filter1", "LCR_label_value": "Filter2|Filter3"})
        result = process_filter_columns(
            row,
            filter_columns_registry=self.test_filter_columns_registry,
            filter_tags_registry=self.test_filter_tag_registry,
        )
        assert compare_strings_ignoring_order(result, "Filter1,Filter2,Filter3")

    def test_real_world_example(self):
        """Test with realistic filter values."""
        row = pd.Series({"jalign_filter": "NO_JUMP_ALIGNMENT", "LCR_label_value": "Coverage-Mappability|Clusters"})
        result = process_filter_columns(
            row,
            filter_columns_registry=self.test_filter_columns_registry,
            filter_tags_registry=self.test_filter_tag_registry,
        )
        assert compare_strings_ignoring_order(result, "NO_JUMP_ALIGNMENT,Coverage-Mappability,Clusters")

    def test_write_combined_vcf(self, tmpdir, resources_dir):
        sample_name = "TEST_HG002_chr19"
        cnv_annotated_bed_file = pjoin(
            resources_dir, "expected_test_HG002.cnmops_cnvpytor.cnvs.combined.bed.annotate.bed"
        )

        fasta_index_file = pjoin(resources_dir, "chr19.fasta.fai")
        outfile = pjoin(tmpdir, f"{sample_name}.cnv.vcf.gz")

        # Use the new workflow - prepare dataframe and write VCF
        cnv_df = convert_combined_cnv_results_to_output_formats.prepare_cnv_dataframe(cnv_annotated_bed_file)
        convert_combined_cnv_results_to_output_formats.write_cnv_vcf(outfile, cnv_df, sample_name, fasta_index_file)

        expected_vcf_file = pjoin(resources_dir, "expected_test_HG002.legacy.cnv.vcf.gz")
        compare_vcfs(expected_vcf_file, outfile)
