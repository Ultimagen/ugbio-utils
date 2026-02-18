import json
from pathlib import Path

import polars as pl
import pytest
from ugbio_featuremap.featuremap_to_dataframe import (
    _apply_read_filters,
    _load_read_filters,
    vcf_to_parquet,
)

# Test data paths
TEST_FEATUREMAP = Path(__file__).parent.parent / "resources" / "23A03846_bc_30.head.featuremap.vcf.gz"
TEST_FEATUREMAP_RANDOM = Path(__file__).parent.parent / "resources" / "23A03846_bc_30.head.random.featuremap.vcf.gz"
TEST_FILTERS_JSON = Path(__file__).parent.parent / "resources" / "create_featuremap_read_filters_test.json"


def test_load_read_filters_real_file():
    """Test loading read filters from the actual JSON file."""
    if not Path(TEST_FILTERS_JSON).exists():
        pytest.skip(f"Test filter file not found: {TEST_FILTERS_JSON}")

    result = _load_read_filters(TEST_FILTERS_JSON)
    assert result is not None
    assert isinstance(result, dict)
    print(f"Loaded filters: {result}")


def test_vcf_to_parquet_with_real_filters(tmp_path: Path):
    """Test full VCF to Parquet conversion with real featuremap and filters."""
    if not Path(TEST_FEATUREMAP).exists():
        pytest.skip(f"Test featuremap file not found: {TEST_FEATUREMAP}")
    if not Path(TEST_FILTERS_JSON).exists():
        pytest.skip(f"Test filter file not found: {TEST_FILTERS_JSON}")

    output_parquet = tmp_path / "filtered_output.parquet"

    # Convert with read filters - should produce valid output
    vcf_to_parquet(
        vcf=TEST_FEATUREMAP,
        out=str(output_parquet),
        drop_format={"GT", "AD", "X_TCM"},
        jobs=2,
        read_filters_json=TEST_FILTERS_JSON,
        chunk_bp=1_000_000,  # Smaller chunks for test
    )

    # Verify the output exists and is not empty
    assert output_parquet.exists(), "Output Parquet file should exist"
    result_df = pl.read_parquet(output_parquet)
    print(f"Output shape: {result_df.shape}")
    print(f"Columns: {result_df.columns}")

    # Verify that we have some reads after filtering (not all filtered out)
    assert result_df.height > 0, "Filtered output should not be empty - all reads were filtered out"
    assert "CHROM" in result_df.columns
    assert "POS" in result_df.columns


def test_vcf_to_parquet_no_filters_comparison(tmp_path: Path):
    """Test VCF to Parquet conversion without filters for comparison."""
    if not Path(TEST_FEATUREMAP).exists():
        pytest.skip(f"Test featuremap file not found: {TEST_FEATUREMAP}")

    output_parquet = tmp_path / "unfiltered_output.parquet"

    # Convert without read filters
    vcf_to_parquet(
        vcf=TEST_FEATUREMAP,
        out=str(output_parquet),
        drop_format={"GT", "AD", "X_TCM"},
        jobs=2,
        chunk_bp=1_000_000,  # Smaller chunks for test
    )

    assert output_parquet.exists()

    # Read the output
    result_df = pl.read_parquet(output_parquet)
    print(f"Unfiltered output shape: {result_df.shape}")

    # Basic sanity checks
    assert result_df.height > 0, "Unfiltered output should not be empty"
    assert "CHROM" in result_df.columns
    assert "POS" in result_df.columns


def test_load_read_filters_with_key(tmp_path: Path):
    """Test loading read filters from JSON with specific key."""
    data = {
        "experiment_1": {"filters": [{"field": "BCSQ", "op": "ge", "value": 20, "type": "quality"}]},
        "experiment_2": {"filters": [{"field": "MAPQ", "op": "ge", "value": 40, "type": "quality"}]},
    }

    json_file = tmp_path / "test_filters_with_keys.json"
    with open(json_file, "w") as f:
        json.dump(data, f)

    result = _load_read_filters(str(json_file), "experiment_1")
    assert result == data["experiment_1"]


def test_load_read_filters_missing_key(tmp_path: Path):
    """Test error handling when key doesn't exist in JSON."""
    data = {"experiment_1": {"filters": []}}

    json_file = tmp_path / "test_filters.json"
    with open(json_file, "w") as f:
        json.dump(data, f)

    with pytest.raises(ValueError, match="Key 'missing_key' not found"):
        _load_read_filters(str(json_file), "missing_key")


def test_load_read_filters_none():
    """Test that None input returns None."""
    result = _load_read_filters(None)
    assert result is None


def test_load_read_filters_invalid_json(tmp_path: Path):
    """Test error handling for invalid JSON."""
    json_file = tmp_path / "invalid.json"
    with open(json_file, "w") as f:
        f.write("invalid json {")

    with pytest.raises(RuntimeError, match="Failed to load read filters"):
        _load_read_filters(str(json_file))


def test_apply_read_filters_empty_dataframe():
    """Test applying filters to an empty dataframe."""
    empty_df = pl.DataFrame({"CHROM": [], "POS": [], "BCSQ": [], "MAPQ": []})

    filters = {"filters": [{"field": "BCSQ", "op": "ge", "value": 30, "type": "quality"}]}

    result = _apply_read_filters(empty_df, filters)
    assert result.height == 0


def test_apply_read_filters_no_filters():
    """Test that no filters returns original dataframe."""
    df_no_filters = pl.DataFrame({"CHROM": ["chr1", "chr2"], "POS": [100, 200], "BCSQ": [25, 35]})

    result = _apply_read_filters(df_no_filters, None)
    assert result.equals(df_no_filters)


def test_apply_read_filters_quality_filter():
    """Test applying a quality filter."""
    df_quality_filters = pl.DataFrame(
        {"CHROM": ["chr1", "chr1", "chr2"], "POS": [100, 150, 200], "BCSQ": [25, 35, 40], "MAPQ": [30, 50, 70]}
    )

    filters = {"filters": [{"field": "BCSQ", "op": "ge", "value": 30, "type": "quality"}]}

    result = _apply_read_filters(df_quality_filters, filters)

    # Should keep only rows where BCSQ >= 30
    expected_height = 2  # rows with BCSQ 35 and 40
    assert result.height == expected_height
    assert all(result["BCSQ"] >= 30)


def test_vcf_to_parquet_with_filter_key(tmp_path: Path):
    """Test VCF to Parquet conversion using a specific filter key."""
    if not Path(TEST_FEATUREMAP_RANDOM).exists():
        pytest.skip(f"Test featuremap file not found: {TEST_FEATUREMAP_RANDOM}")
    if not Path(TEST_FILTERS_JSON).exists():
        pytest.skip(f"Test filter file not found: {TEST_FILTERS_JSON}")

    output_parquet = tmp_path / "filtered_output_f2.parquet"

    # Convert with read filters using 'f2_filters' key - should produce valid output
    vcf_to_parquet(
        vcf=TEST_FEATUREMAP_RANDOM,
        out=str(output_parquet),
        drop_format={"GT", "AD", "X_TCM"},
        jobs=2,
        read_filters_json=TEST_FILTERS_JSON,
        read_filter_json_key="f2_filters",
        chunk_bp=1_000_000,  # Smaller chunks for test
    )

    # Verify the output exists and is not empty
    assert output_parquet.exists(), "Output Parquet file should exist"
    result_df = pl.read_parquet(output_parquet)
    print(f"F2 filtered output shape: {result_df.shape}")

    # Verify that we have some reads after filtering (not all filtered out)
    assert result_df.height > 0, "F2 filtered output should not be empty - all reads were filtered out"
    assert "CHROM" in result_df.columns
    assert "POS" in result_df.columns


def test_vcf_to_parquet_with_f2_filters(tmp_path: Path):
    """Test VCF to Parquet conversion with f2_filters using the random featuremap."""
    f2_input_vcf = "/data/Runs/SRSNV/251222_filters/23A03846_bc_30.head.random.featuremap.vcf.gz"
    read_filters_json = "/data/Runs/SRSNV/251222_filters/create_featuremap_read_filters_test.json"

    if not Path(f2_input_vcf).exists():
        pytest.skip(f"F2 test featuremap file not found: {f2_input_vcf}")
    if not Path(read_filters_json).exists():
        pytest.skip(f"Test filter file not found: {read_filters_json}")

    output_parquet = tmp_path / "f2_filtered_output.parquet"

    # Convert with f2_filters
    vcf_to_parquet(
        vcf=f2_input_vcf,
        out=str(output_parquet),
        drop_format={"GT", "AD", "X_TCM"},
        jobs=2,
        read_filters_json=read_filters_json,
        read_filter_json_key="f2_filters",
        chunk_bp=1_000_000,  # Smaller chunks for test
    )

    assert output_parquet.exists()

    # Read the output and verify it's not empty
    result_df = pl.read_parquet(output_parquet)
    print(f"F2 filters with random featuremap output shape: {result_df.shape}")

    # Basic sanity checks
    assert result_df.height > 0, "F2 filtered output should not be empty"
    assert "CHROM" in result_df.columns
    assert "POS" in result_df.columns


def test_integration_filtering_works(tmp_path: Path):
    """Integration test to verify the filtering pipeline works end-to-end."""
    if not Path(TEST_FEATUREMAP).exists():
        pytest.skip(f"Test featuremap file not found: {TEST_FEATUREMAP}")

    # First convert without filters
    output_unfiltered = tmp_path / "unfiltered_output.parquet"
    vcf_to_parquet(
        vcf=TEST_FEATUREMAP,
        out=str(output_unfiltered),
        drop_format={"GT", "AD", "X_TCM"},
        jobs=1,
        chunk_bp=1_000_000,
    )

    # Then convert with very permissive filters that should pass some reads
    output_filtered = tmp_path / "filtered_output.parquet"
    permissive_filters = tmp_path / "permissive_filters.json"

    # Create a filter that only requires BCSQ > 0 (very permissive)
    filter_config = {"filters": [{"field": "BCSQ", "op": "gt", "value": 0, "type": "quality"}]}

    with open(permissive_filters, "w") as f:
        json.dump(filter_config, f)

    vcf_to_parquet(
        vcf=TEST_FEATUREMAP,
        out=str(output_filtered),
        drop_format={"GT", "AD", "X_TCM"},
        jobs=1,
        read_filters_json=str(permissive_filters),
        chunk_bp=1_000_000,
    )

    # Both should exist and have data
    assert output_unfiltered.exists()
    assert output_filtered.exists()

    unfiltered_df = pl.read_parquet(output_unfiltered)
    filtered_df = pl.read_parquet(output_filtered)

    print(f"Unfiltered: {unfiltered_df.shape}, Filtered: {filtered_df.shape}")

    # Both should have data
    assert unfiltered_df.height > 0
    assert filtered_df.height > 0

    # Filtered should have same or fewer reads
    assert filtered_df.height <= unfiltered_df.height

    # All filtered reads should satisfy the filter condition
    assert all(filtered_df["BCSQ"] > 0)


if __name__ == "__main__":
    pytest.main([__file__])
