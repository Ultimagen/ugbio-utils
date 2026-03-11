"""Shared fixtures and utilities for somatic featuremap classifier tests."""

from pathlib import Path

import polars as pl
import pysam
import pytest

RESOURCES_DIR = Path(__file__).parent / "resources"

# Sample names in the test VCFs
TUMOR_SAMPLE = "Pa_46_FreshFrozen"
NORMAL_SAMPLE = "Pa_46_Buffycoat"


# =============================================================================
# Path fixtures
# =============================================================================
@pytest.fixture
def resources_dir():
    return RESOURCES_DIR


@pytest.fixture
def mini_somatic_vcf():
    """Minimal somatic featuremap VCF (~28 variants) for fast tests.

    Contains:
    - 13 PASS variants (low / medium / high VAF)
    - 10 PreFiltered variants
    - 5 SingleRead variants
    - 1 multi-allelic site (3 records at chr19:1023622)
    - 2 samples: Pa_46_FreshFrozen (tumor), Pa_46_Buffycoat (normal)
    """
    path = RESOURCES_DIR / "somatic_featuremap_input.vcf.gz"
    assert path.exists(), f"Test VCF not found: {path}"
    return path


@pytest.fixture
def tr_bed():
    """Tandem repeat BED for the test region."""
    path = RESOURCES_DIR / "tr_input.bed"
    assert path.exists(), f"TR BED not found: {path}"
    return path


@pytest.fixture
def genome_fai():
    """Reference genome FASTA index (.fai) file."""
    path = RESOURCES_DIR / "Homo_sapiens_assembly38.fasta.fai"
    assert path.exists(), f"Genome FAI not found: {path}"
    return path


@pytest.fixture
def xgb_model_fresh_frozen():
    """XGBoost model V1.15 for Fresh/Frozen samples."""
    path = RESOURCES_DIR / "HG006_HG003.v1.25.WG.t_alt_readsGT1.V1.15.json"
    assert path.exists(), f"XGB model V1.15 not found: {path}"
    return path


@pytest.fixture
def xgb_model_ffpe():
    """XGBoost FFPE model V1.7."""
    path = RESOURCES_DIR / "FFPE_TapasPa46_Pa67.v1.26.WG.t_alt_readsGT1.FFPEV1.7.json"
    assert path.exists(), f"XGB model FFPE not found: {path}"
    return path


@pytest.fixture
def tumor_sample():
    return TUMOR_SAMPLE


@pytest.fixture
def normal_sample():
    return NORMAL_SAMPLE


# =============================================================================
# Validation helpers
# =============================================================================
def validate_output_vcf(
    vcf_path: Path,
    *,
    expected_info_fields: list[str] | None = None,
    expected_format_fields: list[str] | None = None,
    expected_samples: list[str] | None = None,
    min_records: int = 1,
):
    """Validate output VCF structure and content.

    Parameters
    ----------
    vcf_path : Path
        Path to the VCF file.
    expected_info_fields : list[str], optional
        INFO fields that must exist in the header.
    expected_format_fields : list[str], optional
        FORMAT fields that must exist in the header.
    expected_samples : list[str], optional
        Expected sample names in order.
    min_records : int
        Minimum number of variant records expected.
    """
    assert vcf_path.exists(), f"Output VCF does not exist: {vcf_path}"

    with pysam.VariantFile(str(vcf_path)) as vcf:
        if expected_info_fields:
            header_info = set(vcf.header.info.keys())
            missing = set(expected_info_fields) - header_info
            assert not missing, f"Missing INFO fields in VCF header: {sorted(missing)}"

        if expected_format_fields:
            header_formats = set(vcf.header.formats.keys())
            missing = set(expected_format_fields) - header_formats
            assert not missing, f"Missing FORMAT fields in VCF header: {sorted(missing)}"

        if expected_samples:
            assert (
                list(vcf.header.samples) == expected_samples
            ), f"Sample mismatch: expected {expected_samples}, got {list(vcf.header.samples)}"

        records = list(vcf)
        assert len(records) >= min_records, f"Expected at least {min_records} records, got {len(records)}"

    return records


def validate_parquet_schema(
    parquet_path: Path,
    required_columns: set[str],
    *,
    min_rows: int = 1,
) -> pl.DataFrame:
    """Validate Parquet file schema and content.

    Parameters
    ----------
    parquet_path : Path
        Path to the Parquet file.
    required_columns : set[str]
        Column names that must be present.
    min_rows : int
        Minimum number of rows expected.

    Returns
    -------
    pl.DataFrame
        The loaded DataFrame.
    """
    assert parquet_path.exists(), f"Parquet file does not exist: {parquet_path}"

    loaded_df = pl.read_parquet(parquet_path)
    actual_columns = set(loaded_df.columns)
    missing = required_columns - actual_columns
    assert not missing, (
        f"Missing columns in parquet:\n" f"  Missing: {sorted(missing)}\n" f"  Available: {sorted(actual_columns)}"
    )
    assert len(loaded_df) >= min_rows, f"Expected at least {min_rows} rows, got {len(loaded_df)}"

    return loaded_df


def count_vcf_records(vcf_path: Path, filter_string: str | None = None) -> int:
    """Count records in a VCF file, optionally filtering by FILTER field."""
    count = 0
    with pysam.VariantFile(str(vcf_path)) as vcf:
        for rec in vcf:
            if filter_string is None or filter_string in rec.filter:
                count += 1
    return count
