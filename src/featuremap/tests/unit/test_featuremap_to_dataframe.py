from __future__ import annotations

import gzip
import math
import os
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from ugbio_featuremap import featuremap_to_dataframe
from ugbio_featuremap.featuremap_to_dataframe import (
    ALT,
    CHROM,
    POS,
    REF,
    X_ALT,
    _build_explicit_schema,
    _cast_expr,
    _get_awk_script_path,
    _resolve_bcftools_command,
    header_meta,
)


def _assert_df_equal(
    actual: pl.DataFrame, expected: pl.DataFrame, rtol: float = 1e-5, *, check_row_count: bool = True
) -> None:
    """Assert DataFrames match on expected columns, with tolerance for floats.

    Only columns in `expected` are checked. Additional columns in `actual` are ignored.
    """
    # Sort both by POS for consistent comparison
    if "POS" in actual.columns:
        actual = actual.sort("POS")
        expected = expected.sort("POS")

    if check_row_count:
        assert (
            actual.height == expected.height
        ), f"Row count mismatch: actual {actual.height}, expected {expected.height}"

    # Check expected columns exist and values match
    for col in expected.columns:
        assert col in actual.columns, f"Missing expected column: {col}"
        actual_vals = actual[col].to_list()
        expected_vals = expected[col].to_list()

        if actual[col].dtype == pl.Float64:
            for i, (a, e) in enumerate(zip(actual_vals, expected_vals)):
                if e is None or (isinstance(e, float) and math.isnan(e)):
                    assert a is None or (
                        isinstance(a, float) and math.isnan(a)
                    ), f"{col}[{i}]: expected None/NaN, got {a}"
                elif a is None or (isinstance(a, float) and math.isnan(a)):
                    assert False, f"{col}[{i}]: expected {e}, got None/NaN"
                else:
                    assert abs(a - e) < rtol, f"{col}[{i}]: expected {e}, got {a}"
        else:
            assert actual_vals == expected_vals, f"{col}: expected {expected_vals}, got {actual_vals}"


# --- fixtures --------------------------------------------------------------
@pytest.fixture(params=["416119-L7402.raw.featuremap.head.vcf.gz"])
def input_featuremap(request):
    """Return each sample VCF in turn."""
    return Path(__file__).parent.parent / "resources" / request.param


@pytest.fixture
def input_categorical_features():
    return Path(__file__).parent.parent / "resources" / "416119-L7402-Z0296-CATCTATCAGGCGAT.categorical_features.json"


def test_comprehensive_vcf_to_parquet_conversion(tmp_path: Path, input_featuremap: Path) -> None:
    """Full pipeline should yield the correct per-read row count and include key columns."""
    # Allow overriding output directory via environment variable

    output_dir = Path(os.environ.get("TEST_FEATUREMAP_TO_DATAFRAME_DIR", tmp_path))
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = tmp_path

    out_path = str(output_path / input_featuremap.name.replace(".vcf.gz", ".parquet"))
    out_path_2 = str(output_path / input_featuremap.name.replace(".2.vcf.gz", ".parquet"))

    # Capture warnings to ensure no "Dropping list columns with inconsistent length" warning is raised
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # run conversion (drop GT by default)
        featuremap_to_dataframe.vcf_to_parquet(
            vcf=str(input_featuremap),
            out=out_path,
            drop_info=set(),
            drop_format={"GT", "AD"},
            jobs=1,  # Use single job to avoid multi-job hanging issue for now
        )
    # Assert the specific warning was NOT raised
    assert not any(
        "Dropping list columns with inconsistent length" in str(w.message) for w in caught
    ), "Unexpected warning: 'Dropping list columns with inconsistent length'"

    featuremap_dataframe = pl.read_parquet(out_path)
    featuremap_dataframe.write_parquet(out_path_2)

    # hard-coded expected row counts per sample
    expected_rows = {
        "416119-L7402.raw.featuremap.head.vcf.gz": 32947,
    }[input_featuremap.name]
    assert featuremap_dataframe.shape[0] == expected_rows

    # sanity-check a few expected columns and types
    assert {"RN", "RL", "X_PREV1"}.issubset(featuremap_dataframe.columns)
    assert featuremap_dataframe["RN"].dtype == pl.Utf8
    assert featuremap_dataframe["POS"].dtype == pl.Int64


def test_enum_column_is_categorical(tmp_path: Path, input_featuremap: Path) -> None:
    """
    Columns whose description lists {A,C,G,T} should be stored as Enum
    with exactly those four categories plus empty string.
    """
    out_path = str(tmp_path / input_featuremap.name.replace(".vcf.gz", ".parquet"))
    featuremap_to_dataframe.vcf_to_parquet(
        vcf=str(input_featuremap),
        out=out_path,
        drop_info=set(),
        drop_format={"GT", "AD"},
        jobs=1,  # Force single job for test stability
    )

    featuremap_dataframe = pl.read_parquet(out_path)
    print(featuremap_dataframe.schema)
    col = featuremap_dataframe["X_PREV1"]
    # Check that it's an Enum type
    assert isinstance(col.dtype, pl.Enum)

    cats = set(col.cat.get_categories())
    assert cats == {"", "A", "C", "G", "T"}


def test_roundtrip(tmp_path: Path, input_featuremap: Path):
    """Parquet row count == total RN elements in source VCF."""
    out = tmp_path / "out.parquet"
    featuremap_to_dataframe.vcf_to_parquet(
        str(input_featuremap), str(out), drop_info=set(), drop_format={"GT", "AD"}, jobs=1
    )

    featuremap_dataframe = pl.read_parquet(out)

    # count RN elements straight from bcftools (no header confusion)
    rn_bytes = subprocess.check_output(
        ["bcftools", "query", "-f", "[%RN\n]", str(input_featuremap)],
        text=False,
    )
    rn_len = sum(len(line.strip().split(b",")) for line in rn_bytes.splitlines())

    assert featuremap_dataframe.height == rn_len


# ------------- REF/ALT default categories ---------------------------------
def test_ref_alt_defaults(tmp_path: Path, input_featuremap: Path):
    out = tmp_path / "def.parquet"
    featuremap_to_dataframe.vcf_to_parquet(
        str(input_featuremap), str(out), drop_info=set(), drop_format={"GT", "AD"}, jobs=1
    )
    featuremap_dataframe = pl.read_parquet(out)

    # REF includes IUPAC ambiguity codes
    assert set(featuremap_dataframe["REF"].cat.get_categories()) == {
        "",
        "A",
        "C",
        "G",
        "T",
        "R",
        "Y",
        "K",
        "M",
        "S",
        "W",
        "B",
        "D",
        "H",
        "V",
        "N",
    }

    # ALT includes just the four bases (+"")
    assert set(featuremap_dataframe["ALT"].cat.get_categories()) == {
        "",
        "A",
        "C",
        "G",
        "T",
    }


# ------------- tiny unit tests per helper ---------------------------------
def test_enum():
    assert featuremap_to_dataframe._enum("foo {A,B}") == ["A", "B"]
    assert featuremap_to_dataframe._enum("no enum") is None


def test_header_meta(input_featuremap):
    bcftools = featuremap_to_dataframe._resolve_bcftools_command()
    info, fmt = featuremap_to_dataframe.header_meta(str(input_featuremap), bcftools)
    assert "X_PREV1" in info
    assert "RN" in fmt


def test_cast_column_categorical():
    featuremap_dataframe = pl.DataFrame({"x": ["A", None]})
    meta = {"type": "String", "cat": ["A", "B"]}
    featuremap_dataframe_2 = featuremap_dataframe.with_columns(_cast_expr("x", meta))
    assert isinstance(featuremap_dataframe_2["x"].dtype, pl.Enum)
    assert set(featuremap_dataframe_2["x"].cat.get_categories()) == {"", "A", "B"}
    assert featuremap_dataframe_2["x"].null_count() == 0


def test_selected_dtypes(tmp_path: Path, input_featuremap: Path):
    out = tmp_path / "full.parquet"
    featuremap_to_dataframe.vcf_to_parquet(
        str(input_featuremap), str(out), drop_info=set(), drop_format={"GT", "AD"}, jobs=1
    )

    featuremap_dataframe = pl.read_parquet(out)

    # Check specific column types
    assert featuremap_dataframe["CHROM"].dtype == pl.Utf8
    assert featuremap_dataframe["POS"].dtype == pl.Int64
    # REF and ALT are Enum types
    assert isinstance(featuremap_dataframe["REF"].dtype, pl.Enum)
    assert isinstance(featuremap_dataframe["ALT"].dtype, pl.Enum)
    assert featuremap_dataframe["VAF"].dtype == pl.Float64
    assert featuremap_dataframe["RN"].dtype == pl.Utf8

    # -------- ensure no nulls remain in numeric or Enum columns ----------
    # polars.NUMERIC_DTYPES is deprecated – use selectors.numeric()
    numeric_cols = set(featuremap_dataframe.select(pl.selectors.numeric()).columns)
    enum_cols = {c for c, dt in featuremap_dataframe.schema.items() if isinstance(dt, pl.Enum)}
    cols_to_check = list(numeric_cols | enum_cols)
    if cols_to_check:  # defensive – some tiny frames may lack numeric/Enum cols
        assert (
            featuremap_dataframe.select(pl.col(cols_to_check).null_count()).sum().row(0)[0] == 0
        ), "Unexpected null values in numeric / Enum columns"


def test_parallel_vcf_conversion_comprehensive(tmp_path: Path, input_featuremap: Path) -> None:
    """Test the parallel VCF to Parquet conversion produces correct results."""
    parallel_out = str(tmp_path / "parallel_output.parquet")
    sequential_out = str(tmp_path / "sequential_output.parquet")

    # Run parallel conversion
    featuremap_to_dataframe.vcf_to_parquet(
        vcf=str(input_featuremap),
        out=parallel_out,
        drop_info=set(),
        drop_format={"GT", "AD"},
        jobs=2,  # Limited jobs for testing
    )

    # Run another parallel conversion for comparison
    featuremap_to_dataframe.vcf_to_parquet(
        vcf=str(input_featuremap),
        out=sequential_out,
        drop_info=set(),
        drop_format={"GT", "AD"},
        jobs=1,  # Single job for comparison
    )

    # Load both results
    parallel_df = pl.read_parquet(parallel_out)
    sequential_df = pl.read_parquet(sequential_out)

    # Convert to pandas to leverage robust comparison utilities
    pd_parallel = parallel_df.to_pandas()
    pd_sequential = sequential_df.to_pandas()

    # Ensure identical column order
    assert list(pd_parallel.columns) == list(
        pd_sequential.columns
    ), "Column ordering differs between parallel and sequential results"

    # Exact match for non-float columns; tolerant match for float columns
    for col in pd_parallel.columns:
        if pd.api.types.is_float_dtype(pd_parallel[col]):
            assert np.allclose(
                pd_parallel[col].values,
                pd_sequential[col].values,
                rtol=1e-6,
                atol=1e-8,
                equal_nan=True,
            ), f"Float column '{col}' differs between parallel and sequential outputs"
        else:
            assert pd_parallel[col].equals(
                pd_sequential[col]
            ), f"Column '{col}' differs between parallel and sequential outputs"


def test_bcftools_awk_pipeline_chunk_creation(tmp_path: Path, input_featuremap: Path) -> None:
    """Test that the bcftools + AWK + split pipeline creates chunk files correctly."""

    chunk_dir = tmp_path / "chunks"
    chunk_dir.mkdir()

    # Get AWK script path
    awk_script = Path(__file__).parent.parent.parent / "ugbio_featuremap" / "explode_lists.awk"
    assert awk_script.exists(), f"AWK script not found: {awk_script}"

    # Build command like the actual function does
    chunk_prefix = str(chunk_dir / "chunk_")
    chunk_size = 500

    cmd = [
        "bash",
        "-c",
        f"bcftools query -f '%CHROM\\t%POS\\t%ID\\t%REF\\t%ALT\\t%QUAL\\t%FILTER\\t%INFO\\t%FORMAT[\\t%SAMPLE]\\n' "
        f"'{input_featuremap}' | "
        f"awk -f '{awk_script}' | "
        f"split -l {chunk_size} --numeric-suffixes=0 -a 6 - '{chunk_prefix}'",
    ]

    # Run the pipeline
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, check=False)
    assert result.returncode == 0, f"Pipeline failed: {result.stderr}"

    # Check that chunk files were created
    chunks = list(chunk_dir.glob("chunk_*"))
    assert len(chunks) > 0, "No chunk files were created"

    # Check that chunk files have content
    for chunk in chunks:
        assert chunk.stat().st_size > 0, f"Chunk {chunk.name} is empty"

    # Check first chunk has expected format (tab-separated values)
    first_chunk = chunks[0]
    content = first_chunk.read_text()
    lines = content.strip().split("\n")
    assert len(lines) > 0, "First chunk has no lines"

    # Each line should have multiple tab-separated fields
    first_line = lines[0]
    fields = first_line.split("\t")
    assert len(fields) >= 10, f"Expected at least 10 fields, got {len(fields)}"


def test_chunk_processing_tsv_to_parquet_conversion(tmp_path: Path) -> None:
    """Test basic chunk processing functionality with minimal data."""

    # Create a simple test chunk file
    chunk_file = tmp_path / "test_chunk.tsv"
    chunk_file.write_text("chr1\t100\t.\tA\tT\t30\tPASS\t.\tGT\t0/1\n")

    output_file = tmp_path / "output.parquet"

    # Test basic TSV reading and Parquet writing
    cols = [
        "CHROM",
        "POS",
        "ID",
        "REF",
        "ALT",
        "QUAL",
        "FILTER",
        "INFO",
        "FORMAT",
        "SAMPLE1",
    ]

    # Read the chunk
    chunk_df = pl.read_csv(
        str(chunk_file),
        separator="\t",
        has_header=False,
        new_columns=cols,
        null_values=["."],
    )

    assert chunk_df.height == 1, "Should read exactly 1 row"
    assert chunk_df.width == len(cols), f"Should have {len(cols)} columns"

    # Write to Parquet
    chunk_df.write_parquet(str(output_file))

    # Verify written file
    assert output_file.exists(), "Output Parquet file should exist"

    # Read back and verify
    verification_df = pl.read_parquet(str(output_file))
    assert verification_df.shape == chunk_df.shape, "Round-trip should preserve shape"
    assert verification_df.columns == chunk_df.columns, "Round-trip should preserve columns"


def test_awk_script_path_resolution() -> None:
    """Test that the AWK script can be found in both development and installed environments."""
    # Test that the function returns a valid path
    awk_path_str = _get_awk_script_path()
    awk_path = Path(awk_path_str)
    assert awk_path.exists(), f"AWK script not found at: {awk_path_str}"
    assert awk_path.name == "explode_lists.awk", f"Wrong filename: {awk_path.name}"
    assert awk_path.is_file(), f"AWK script path is not a file: {awk_path_str}"


def test_decimal_type_inference_and_error_propagation_comprehensive(
    tmp_path: Path,
) -> None:
    """
    Test the complete fix for decimal values being incorrectly parsed as integers.

    This test verifies:
    1. Explicit schema correctly maps VCF types to Polars types
    2. Decimal values like 0.0227273 are parsed as Float64, not i64
    3. The complete pipeline from VCF to Parquet preserves float precision

    This is a comprehensive test for the main issues that were fixed:
    - Schema inference problems causing type mismatches
    """
    # Test VCF with the exact problematic float values from the original issue
    test_vcf_content = """##fileformat=VCFv4.2
##contig=<ID=chr1,length=1000000>
##INFO=<ID=AF,Number=1,Type=Float,Description="Allele Frequency">
##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=VAF,Number=1,Type=Float,Description="Variant Allele Frequency">
##FORMAT=<ID=RN,Number=.,Type=String,Description="Read Names">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	SAMPLE1
chr1	100	.	A	G	30.0	PASS	AF=0.0227273;DP=44	GT:VAF:RN	0/1:0.0227273:read1,read2
chr1	200	.	C	T	25.5	PASS	AF=0.5454545;DP=22	GT:VAF:RN	0/1:0.5454545:read3,read4,read5
"""

    plain = tmp_path / "test_decimal_fix.vcf"
    plain.write_text(test_vcf_content)
    vcf_file = tmp_path / "test_decimal_fix.vcf.gz"
    subprocess.run(
        [
            "bcftools",
            "view",
            str(plain),
            "-Oz",
            "-o",
            str(vcf_file),
            "--write-index=tbi",
        ],
        check=True,
    )
    parquet_file = tmp_path / "test_decimal_fix.parquet"

    bcftools = _resolve_bcftools_command()
    info_meta, fmt_meta = header_meta(str(vcf_file), bcftools)

    # Verify header metadata parsing
    assert info_meta["AF"]["type"] == "Float"
    assert info_meta["DP"]["type"] == "Integer"
    assert fmt_meta["VAF"]["type"] == "Float"
    assert fmt_meta["RN"]["type"] == "String"

    # Test explicit schema generation
    test_cols = [CHROM, POS, REF, ALT, "AF", "DP", "VAF", "RN"]
    schema = _build_explicit_schema(test_cols, info_meta, fmt_meta)

    # Verify correct type mapping
    assert schema["AF"] == pl.Float64
    assert schema["VAF"] == pl.Float64
    assert schema["DP"] == pl.Int64
    assert schema["RN"] == pl.Utf8
    assert schema[POS] == pl.Int64
    assert schema[CHROM] == pl.Utf8

    # Part 2: Test end-to-end VCF to Parquet conversion
    featuremap_to_dataframe.vcf_to_parquet(
        vcf=str(vcf_file),
        out=str(parquet_file),
        drop_info=set(),
        drop_format={"GT"},
        jobs=1,
    )

    # Verify successful conversion and correct types
    result_df = pl.read_parquet(parquet_file)

    # Check data types are correct
    assert result_df["AF"].dtype == pl.Float64
    assert result_df["VAF"].dtype == pl.Float64
    assert result_df["DP"].dtype == pl.Int64
    assert result_df["RN"].dtype == pl.Utf8

    # Check that the problematic decimal values are correctly preserved
    af_values = result_df["AF"].to_list()
    vaf_values = result_df["VAF"].to_list()

    # Original problematic value that was causing failures
    problematic_decimal = 0.0227273

    # Verify the decimal values are parsed correctly (not as integers)
    assert abs(af_values[0] - problematic_decimal) < 1e-6
    assert abs(vaf_values[0] - problematic_decimal) < 1e-6

    print("✅ All decimal value float type inference fixes verified successfully")


def test_single_job_vcf_to_parquet_conversion(tmp_path: Path, input_featuremap: Path) -> None:
    """Test single job processing (jobs=1) works correctly without data loss."""
    out_path = str(tmp_path / input_featuremap.name.replace(".vcf.gz", "_single_job.parquet"))

    # Run conversion with single job (no region splitting)
    featuremap_to_dataframe.vcf_to_parquet(
        vcf=str(input_featuremap),
        out=out_path,
        drop_info=set(),
        drop_format={"GT", "AD"},
        jobs=1,  # Single job processing
    )

    featuremap_dataframe = pl.read_parquet(out_path)

    # Check expected row count for single job processing
    expected_rows = {
        "416119-L7402.raw.featuremap.head.vcf.gz": 32947,
    }[input_featuremap.name]

    assert (
        featuremap_dataframe.shape[0] == expected_rows
    ), f"Expected {expected_rows} rows, got {featuremap_dataframe.shape[0]}"

    # Verify key columns and types
    assert {"RN", "RL", "X_PREV1"}.issubset(featuremap_dataframe.columns)
    assert featuremap_dataframe["RN"].dtype == pl.Utf8
    assert featuremap_dataframe["POS"].dtype == pl.Int64


def test_vcf_requires_index(tmp_path: Path) -> None:
    """vcf_to_parquet should fail fast when the VCF/BCF has no accompanying index."""
    # Minimal one-record VCF (un-indexed)
    vcf_text = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=1000>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t1\t.\tA\tC\t.\tPASS\t.\n"
    )
    gz = tmp_path / "no_index.vcf.gz"

    with gzip.open(gz, "wt") as fh:
        fh.write(vcf_text)  # bgzip not required – we only test the index check

    with pytest.raises(RuntimeError, match="index not found"):
        featuremap_to_dataframe.vcf_to_parquet(str(gz), str(tmp_path / "out.parquet"), jobs=1)


def test_vcf_requires_samples(tmp_path: Path) -> None:
    """vcf_to_parquet should raise an error when the VCF contains no samples."""
    # VCF with variant records but no sample columns
    vcf_text = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=1000>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t1\t.\tA\tC\t30.0\tPASS\t.\n"
        "chr1\t100\t.\tG\tT\t25.0\tPASS\t.\n"
    )
    plain = tmp_path / "no_samples.vcf"
    plain.write_text(vcf_text)
    vcf_gz = tmp_path / "no_samples.vcf.gz"
    subprocess.run(
        ["bcftools", "view", str(plain), "-Oz", "-o", str(vcf_gz), "--write-index=tbi"],
        check=True,
    )

    with pytest.raises(ValueError, match="contains no samples"):
        featuremap_to_dataframe.vcf_to_parquet(str(vcf_gz), str(tmp_path / "out.parquet"), jobs=1)


def test_st_et_are_categorical(tmp_path: Path, input_featuremap: Path) -> None:
    """Columns advertised as enums in the header (e.g. st / et) must be Enum types."""
    out = tmp_path / "enum.parquet"
    featuremap_to_dataframe.vcf_to_parquet(
        str(input_featuremap), str(out), drop_info=set(), drop_format={"GT", "AD"}, jobs=1
    )

    featuremap_dataframe = pl.read_parquet(out)
    # Some files may use upper- or lower-case; check whichever exists
    for tag in ("st", "et"):
        assert isinstance(
            featuremap_dataframe[tag].dtype, pl.Enum
        ), f"{tag} should be Enum type, got {featuremap_dataframe[tag].dtype}"


def test_qual_dtype_float_even_if_empty(tmp_path: Path) -> None:
    """QUAL column should be Float64 even when every value is '.'."""
    vcf_txt = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=1000>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\n"
        "chr1\t10\t.\tA\tT\t.\tPASS\t.\tGT\t0/1\n"
        "chr1\t300\t.\tG\tC\t21.0\tPASS\t.\tGT:DP:AD:RL\t0/1:41:.:.\n"
    )
    plain = tmp_path / "qual_missing.vcf"
    plain.write_text(vcf_txt)
    vcf_gz = tmp_path / "qual_missing.vcf.gz"
    subprocess.run(
        ["bcftools", "view", str(plain), "-Oz", "-o", str(vcf_gz), "--write-index=tbi"],
        check=True,
    )
    out = tmp_path / "out.parquet"
    featuremap_to_dataframe.vcf_to_parquet(str(vcf_gz), str(out), jobs=1)

    featuremap_dataframe = pl.read_parquet(out)
    print(featuremap_dataframe.schema)
    assert featuremap_dataframe["QUAL"].dtype == pl.Float64, "QUAL should be Float64 even if all values are missing"
    assert featuremap_dataframe["QUAL"].to_list() == [0.0, 21.0]


def test_missing_sample_data(tmp_path: Path) -> None:
    """VCF should handle missing sample data ('.' in sample column) correctly."""
    vcf_txt = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=1000>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=VAF,Number=1,Type=Float,Description="Variant Allele Frequency">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\n"
        "chr1\t10\t.\tA\tT\t30.0\tPASS\t.\tGT:VAF\t0/1:0.5\n"
        "chr1\t20\t.\tC\tG\t25.0\tPASS\t.\tGT:VAF\t.:.\n"
    )
    plain = tmp_path / "missing_sample.vcf"
    plain.write_text(vcf_txt)
    vcf_gz = tmp_path / "missing_sample.vcf.gz"
    subprocess.run(
        ["bcftools", "view", str(plain), "-Oz", "-o", str(vcf_gz), "--write-index=tbi"],
        check=True,
    )
    out = tmp_path / "out.parquet"
    featuremap_to_dataframe.vcf_to_parquet(str(vcf_gz), str(out), jobs=1)

    featuremap_dataframe = pl.read_parquet(out)
    # Should have 2 rows (one per variant)
    assert featuremap_dataframe.height == 2, "Should have 2 rows"
    # Check that missing data is handled (VAF should be Float64, filled with 0.0 for missing)
    assert featuremap_dataframe["VAF"].dtype == pl.Float64, "VAF should be Float64"
    # First row should have VAF=0.5, second row should have VAF=0.0 (missing values filled)
    vaf_values = featuremap_dataframe["VAF"].to_list()
    assert vaf_values[0] == 0.5, "First row should have VAF=0.5"
    assert vaf_values[1] == 0.0, "Second row should have VAF=0.0 (missing value filled)"


def test_multi_sample_vcf(tmp_path: Path) -> None:
    """Multi-sample VCF should produce columns with sample name suffixes for FORMAT fields."""
    vcf_txt = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr19,length=58617616>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=VAF,Number=1,Type=Float,Description="Variant Allele Frequency">\n'
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Number of reads containing this location">\n'
        '##FORMAT=<ID=RN,Number=.,Type=String,Description="Query (read) name">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\tSAMPLE2\n"
        "chr19\t271215\t.\tG\tC\t41.94\tPASS\t.\tGT:VAF:DP:RN\t./.:0.0298507:67:021152_1-Z0115-2981480323\t./.:0:44:.\n"
        "chr19\t271241\t.\tA\tG\t42.6\tPASS\t.\tGT:VAF:DP:RN\t./.:0.0151515:66:021152_1-Z0115-2353376084\t./.:0:35:.\n"
    )
    plain = tmp_path / "multi_sample.vcf"
    plain.write_text(vcf_txt)
    vcf_gz = tmp_path / "multi_sample.vcf.gz"
    subprocess.run(
        ["bcftools", "view", str(plain), "-Oz", "-o", str(vcf_gz), "--write-index=tbi"],
        check=True,
    )
    out = tmp_path / "out.parquet"
    featuremap_to_dataframe.vcf_to_parquet(str(vcf_gz), str(out), jobs=1)

    featuremap_dataframe = pl.read_parquet(out)
    # Should have rows (one per read due to RN list explosion)
    assert featuremap_dataframe.height > 0, "Should have at least one row"
    # Check that FORMAT fields are suffixed with sample names
    assert "VAF_SAMPLE1" in featuremap_dataframe.columns, "Should have VAF_SAMPLE1 column"
    assert "VAF_SAMPLE2" in featuremap_dataframe.columns, "Should have VAF_SAMPLE2 column"
    assert "DP_SAMPLE1" in featuremap_dataframe.columns, "Should have DP_SAMPLE1 column"
    assert "DP_SAMPLE2" in featuremap_dataframe.columns, "Should have DP_SAMPLE2 column"
    assert "RN_SAMPLE1" in featuremap_dataframe.columns, "Should have RN_SAMPLE1 column"
    assert "RN_SAMPLE2" in featuremap_dataframe.columns, "Should have RN_SAMPLE2 column"
    # Fixed columns should not be prefixed
    assert "CHROM" in featuremap_dataframe.columns, "CHROM should not be prefixed"
    assert "POS" in featuremap_dataframe.columns, "POS should not be prefixed"
    assert "QUAL" in featuremap_dataframe.columns, "QUAL should not be prefixed"
    # Check that data is correctly joined - both samples should have data for the same positions
    assert featuremap_dataframe["POS"].n_unique() == 2, "Should have 2 unique positions"
    # Check that VAF values are correctly preserved
    sample1_vaf = featuremap_dataframe["VAF_SAMPLE1"].to_list()
    sample2_vaf = featuremap_dataframe["VAF_SAMPLE2"].to_list()
    # At least one row should have the expected VAF values
    assert any(v == 0.0298507 for v in sample1_vaf if v is not None), "Should have VAF=0.0298507 for SAMPLE1"
    assert any(v == 0.0 for v in sample2_vaf if v is not None), "Should have VAF=0.0 for SAMPLE2"


def test_multi_sample_vcf_schema_consistency_single_sample_regions(tmp_path: Path) -> None:
    """
    Test that multi-sample VCFs maintain consistent schema across regions.

    This test verifies the fix for schema inconsistency where regions with only
    one sample having variants would produce columns without sample prefixes,
    causing schema mismatches when merging parquet files.

    The test creates a multi-sample VCF where:
    - Some regions have variants only in SAMPLE1
    - Some regions have variants only in SAMPLE2
    - Some regions have variants in both samples

    All regions should produce columns with sample prefixes (e.g., VAF_SAMPLE1, VAF_SAMPLE2)
    to ensure schema consistency when merging.
    """
    vcf_txt = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=1000000>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=VAF,Number=1,Type=Float,Description="Variant Allele Frequency">\n'
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\tSAMPLE2\n"
        # Region 1: Only SAMPLE1 has variants (chr1:100-5000)
        "chr1\t100\t.\tA\tG\t30.0\tPASS\t.\tGT:VAF:DP\t0/1:0.5:50\t./.:.:.\n"
        "chr1\t200\t.\tC\tT\t25.0\tPASS\t.\tGT:VAF:DP\t0/1:0.3:40\t./.:.:.\n"
        # Region 2: Only SAMPLE2 has variants (chr1:10000-15000)
        "chr1\t10000\t.\tG\tA\t35.0\tPASS\t.\tGT:VAF:DP\t./.:.:.\t0/1:0.6:60\n"
        "chr1\t11000\t.\tT\tC\t28.0\tPASS\t.\tGT:VAF:DP\t./.:.:.\t0/1:0.4:45\n"
        # Region 3: Both samples have variants (chr1:50000-55000)
        "chr1\t50000\t.\tA\tT\t40.0\tPASS\t.\tGT:VAF:DP\t0/1:0.7:70\t0/1:0.2:25\n"
        "chr1\t51000\t.\tC\tG\t32.0\tPASS\t.\tGT:VAF:DP\t0/1:0.8:80\t0/1:0.3:30\n"
    )
    plain = tmp_path / "multi_sample_regions.vcf"
    plain.write_text(vcf_txt)
    vcf_gz = tmp_path / "multi_sample_regions.vcf.gz"
    subprocess.run(
        ["bcftools", "view", str(plain), "-Oz", "-o", str(vcf_gz), "--write-index=tbi"],
        check=True,
    )
    out = tmp_path / "out_regions.parquet"
    # Use small chunk size to force region splitting, testing the edge case
    featuremap_to_dataframe.vcf_to_parquet(str(vcf_gz), str(out), jobs=1, chunk_bp=20000)

    featuremap_dataframe = pl.read_parquet(out)
    # Should have rows
    assert featuremap_dataframe.height > 0, "Should have at least one row"

    # CRITICAL: All regions should produce columns with sample prefixes, even regions
    # where only one sample has variants. This ensures schema consistency when merging.
    assert (
        "VAF_SAMPLE1" in featuremap_dataframe.columns
    ), "Should have VAF_SAMPLE1 column even in regions where only SAMPLE2 has variants"
    assert (
        "VAF_SAMPLE2" in featuremap_dataframe.columns
    ), "Should have VAF_SAMPLE2 column even in regions where only SAMPLE1 has variants"
    assert "DP_SAMPLE1" in featuremap_dataframe.columns, "Should have DP_SAMPLE1 column"
    assert "DP_SAMPLE2" in featuremap_dataframe.columns, "Should have DP_SAMPLE2 column"

    # Verify that columns without prefixes should NOT exist (schema consistency)
    assert (
        "VAF" not in featuremap_dataframe.columns
    ), "VAF column without sample prefix should not exist in multi-sample VCF"
    assert (
        "DP" not in featuremap_dataframe.columns
    ), "DP column without sample prefix should not exist in multi-sample VCF"

    # Verify data correctness:
    # - Regions with only SAMPLE1 should have nulls for SAMPLE2 columns
    # - Regions with only SAMPLE2 should have nulls for SAMPLE1 columns
    # - Regions with both should have values for both

    # Check that we have variants from all three regions
    positions = featuremap_dataframe["POS"].unique().to_list()
    assert 100 in positions or 200 in positions, "Should have variants from region 1 (SAMPLE1 only)"
    assert 10000 in positions or 11000 in positions, "Should have variants from region 2 (SAMPLE2 only)"
    assert 50000 in positions or 51000 in positions, "Should have variants from region 3 (both samples)"

    # Verify that regions with only one sample have missing/default values (0.0) for the other sample
    # Note: Missing Float values are filled with 0.0 by the casting logic
    sample1_only_rows = featuremap_dataframe.filter(pl.col("POS").is_in([100, 200]))
    if sample1_only_rows.height > 0:
        # SAMPLE2 columns should have 0.0 (missing values) for these rows
        sample2_vaf_values = sample1_only_rows["VAF_SAMPLE2"].to_list()
        assert all(
            v == 0.0 for v in sample2_vaf_values
        ), "SAMPLE2 VAF should be 0.0 (missing) in regions where only SAMPLE1 has variants"
        # SAMPLE1 should have actual values
        sample1_vaf_values = sample1_only_rows["VAF_SAMPLE1"].to_list()
        assert any(
            v > 0.0 for v in sample1_vaf_values
        ), "SAMPLE1 should have non-zero VAF values in regions where it has variants"

    sample2_only_rows = featuremap_dataframe.filter(pl.col("POS").is_in([10000, 11000]))
    if sample2_only_rows.height > 0:
        # SAMPLE1 columns should have 0.0 (missing values) for these rows
        sample1_vaf_values = sample2_only_rows["VAF_SAMPLE1"].to_list()
        assert all(
            v == 0.0 for v in sample1_vaf_values
        ), "SAMPLE1 VAF should be 0.0 (missing) in regions where only SAMPLE2 has variants"
        # SAMPLE2 should have actual values
        sample2_vaf_values = sample2_only_rows["VAF_SAMPLE2"].to_list()
        assert any(
            v > 0.0 for v in sample2_vaf_values
        ), "SAMPLE2 should have non-zero VAF values in regions where it has variants"

    # Verify that regions with both samples have values for both
    both_samples_rows = featuremap_dataframe.filter(pl.col("POS").is_in([50000, 51000]))
    if both_samples_rows.height > 0:
        sample1_vaf_both = both_samples_rows["VAF_SAMPLE1"].drop_nulls().to_list()
        sample2_vaf_both = both_samples_rows["VAF_SAMPLE2"].drop_nulls().to_list()
        assert len(sample1_vaf_both) > 0, "SAMPLE1 should have VAF values in regions with both samples"
        assert len(sample2_vaf_both) > 0, "SAMPLE2 should have VAF values in regions with both samples"


def test_x_alt_categories(tmp_path: Path, input_featuremap: Path) -> None:
    """
    X_ALT (alternative REF for reverse-complement) must be Enum with the same
    category dictionary as REF.
    """
    out = tmp_path / "xalt.parquet"
    featuremap_to_dataframe.vcf_to_parquet(
        str(input_featuremap), str(out), drop_info=set(), drop_format={"GT", "AD"}, jobs=1
    )
    frame = pl.read_parquet(out)

    # Only run the assertion when the column exists in the file
    if X_ALT in frame.columns:
        assert isinstance(frame[X_ALT].dtype, pl.Enum), "X_ALT should be Enum"
        assert set(frame[X_ALT].cat.get_categories()) == set(
            frame[ALT].cat.get_categories()
        ), "X_ALT categories must match ALT categories"


def test_aggregate_mode_list_fields(tmp_path: Path) -> None:
    """
    Test that aggregate mode replaces list columns with aggregation metrics (mean, min, max, count, count_zero).

    This test verifies:
    1. List columns are replaced with 5 columns (mean, min, max, count, count_zero)
    2. Row count is one per variant (not exploded)
    3. Aggregate columns have correct types (Float64 for mean/min/max, Int64 for count/count_zero)
    4. Aggregate values are computed correctly
    """
    # Create a VCF with list fields containing numeric values (including zeros)
    vcf_txt = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=1000000>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
        '##FORMAT=<ID=AD,Number=.,Type=Integer,Description="Allelic depths">\n'
        '##FORMAT=<ID=RL,Number=.,Type=Integer,Description="Read lengths">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\n"
        "chr1\t100\t.\tA\tG\t30.0\tPASS\t.\tGT:DP:AD:RL\t0/1:50:10,0,15:100,0,102\n"
        "chr1\t200\t.\tC\tT\t25.0\tPASS\t.\tGT:DP:AD:RL\t0/1:30:0,0:200,201\n"
        "chr1\t300\t.\tG\tA\t20.0\tPASS\t.\tGT:DP:AD:RL\t0/1:40:.:.\n"
    )
    plain = tmp_path / "aggregate_test.vcf"
    plain.write_text(vcf_txt)
    vcf_gz = tmp_path / "aggregate_test.vcf.gz"
    subprocess.run(
        ["bcftools", "view", str(plain), "-Oz", "-o", str(vcf_gz), "--write-index=tbi"],
        check=True,
    )

    # Test aggregate mode
    out_aggregate = tmp_path / "aggregate.parquet"
    featuremap_to_dataframe.vcf_to_parquet(
        str(vcf_gz), str(out_aggregate), drop_format={"GT"}, list_mode="aggregate", jobs=1
    )

    aggregate_df = pl.read_parquet(out_aggregate)

    # Expected output: aggregate mode should produce one row per variant with aggregated metrics
    # Variant 1 (POS=100): AD=[10,0,15] -> mean=8.33, min=0, max=15, count=3, count_zero=1
    #                      RL=[100,0,102] -> mean=67.33, min=0, max=102, count=3, count_zero=1
    # Variant 2 (POS=200): AD=[0,0] -> mean=0.0, min=0, max=0, count=2, count_zero=2
    #                      RL=[200,201] -> mean=200.5, min=200, max=201, count=2, count_zero=0
    # Variant 3 (POS=300): AD=[.] -> mean=None, min=None, max=None, count=0, count_zero=0
    #                      RL=[.] -> mean=None, min=None, max=None, count=0, count_zero=0
    expected = pl.DataFrame(
        {
            "CHROM": ["chr1", "chr1", "chr1"],
            "POS": [100, 200, 300],
            "QUAL": [30.0, 25.0, 20.0],
            "REF": ["A", "C", "G"],
            "ALT": ["G", "T", "A"],
            "DP": [50, 30, 40],
            "AD_mean": [25.0 / 3.0, 0.0, None],
            "AD_min": [0.0, 0.0, None],
            "AD_max": [15.0, 0.0, None],
            "AD_count": [3, 2, 0],
            "AD_count_zero": [1, 2, 0],
            "RL_mean": [202.0 / 3.0, 200.5, None],
            "RL_min": [0.0, 200.0, None],
            "RL_max": [102.0, 201.0, None],
            "RL_count": [3, 2, 0],
            "RL_count_zero": [1, 0, 0],
        }
    )

    _assert_df_equal(aggregate_df, expected)
    assert (
        "AD" not in aggregate_df.columns and "RL" not in aggregate_df.columns
    ), "Original list columns should be removed"


def test_aggregate_mode_multi_sample_vcf(tmp_path: Path) -> None:
    """
    Test that aggregate mode works correctly for multi-sample VCFs.

    This test verifies:
    1. List columns are replaced with aggregate metrics (mean, min, max, count, count_zero)
    2. Row count is one per variant (not exploded)
    3. Aggregate values are computed correctly for each sample independently
    4. Aggregate columns get sample suffixes (e.g., AD_mean_SAMPLE1, AD_mean_SAMPLE2)
    5. Non-FORMAT columns (QUAL, INFO) are not duplicated
    """
    # Create a multi-sample VCF with list fields
    vcf_txt = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=1000000>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
        '##FORMAT=<ID=AD,Number=.,Type=Integer,Description="Allelic depths">\n'
        '##FORMAT=<ID=RL,Number=.,Type=Integer,Description="Read lengths">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\tSAMPLE2\n"
        "chr1\t100\t.\tA\tG\t30.0\tPASS\t.\tGT:DP:AD:RL\t0/1:50:10,0,15:100,101,102\t0/1:40:0,10:200,201\n"
        "chr1\t200\t.\tC\tT\t25.0\tPASS\t.\tGT:DP:AD:RL\t0/1:30:8,12:150,151\t0/1:35:.:.\n"
    )
    plain = tmp_path / "multi_sample_aggregate.vcf"
    plain.write_text(vcf_txt)
    vcf_gz = tmp_path / "multi_sample_aggregate.vcf.gz"
    subprocess.run(
        ["bcftools", "view", str(plain), "-Oz", "-o", str(vcf_gz), "--write-index=tbi"],
        check=True,
    )

    # Test aggregate mode
    out_aggregate = tmp_path / "multi_aggregate.parquet"
    featuremap_to_dataframe.vcf_to_parquet(
        str(vcf_gz), str(out_aggregate), drop_format={"GT"}, list_mode="aggregate", jobs=1
    )

    multi_aggregate_df = pl.read_parquet(out_aggregate)

    # Expected output: one row per variant with aggregated metrics per sample
    # Aggregate columns get sample suffixes like regular FORMAT columns
    # Variant 1 (POS=100):
    #   SAMPLE1: AD=[10,0,15] -> mean=8.33, min=0, max=15, count=3, count_zero=1
    #            RL=[100,101,102] -> mean=101.0, min=100, max=102, count=3, count_zero=0
    #   SAMPLE2: AD=[0,10] -> mean=5.0, min=0, max=10, count=2, count_zero=1
    #            RL=[200,201] -> mean=200.5, min=200, max=201, count=2, count_zero=0
    # Variant 2 (POS=200):
    #   SAMPLE1: AD=[8,12] -> mean=10.0, min=8, max=12, count=2, count_zero=0
    #            RL=[150,151] -> mean=150.5, min=150, max=151, count=2, count_zero=0
    #   SAMPLE2: AD=[.] -> mean=None, min=None, max=None, count=0, count_zero=0
    #            RL=[.] -> mean=None, min=None, max=None, count=0, count_zero=0
    expected = pl.DataFrame(
        {
            "POS": [100, 200],
            "DP_SAMPLE1": [50, 30],
            "DP_SAMPLE2": [40, 35],
            "AD_mean_SAMPLE1": [25.0 / 3.0, 10.0],
            "AD_min_SAMPLE1": [0.0, 8.0],
            "AD_max_SAMPLE1": [15.0, 12.0],
            "AD_count_SAMPLE1": [3, 2],
            "AD_count_zero_SAMPLE1": [1, 0],
            "RL_mean_SAMPLE1": [101.0, 150.5],
            "RL_count_SAMPLE1": [3, 2],
            "RL_count_zero_SAMPLE1": [0, 0],
            "AD_mean_SAMPLE2": [5.0, None],
            "AD_min_SAMPLE2": [0.0, None],
            "AD_max_SAMPLE2": [10.0, None],
            "AD_count_SAMPLE2": [2, 0],
            "AD_count_zero_SAMPLE2": [1, 0],
            "RL_mean_SAMPLE2": [200.5, None],
            "RL_count_SAMPLE2": [2, 0],
            "RL_count_zero_SAMPLE2": [0, 0],
        }
    )
    _assert_df_equal(multi_aggregate_df, expected)

    # Verify aggregate columns have sample suffixes
    assert "AD_mean_SAMPLE1" in multi_aggregate_df.columns and "AD_mean_SAMPLE2" in multi_aggregate_df.columns


def test_aggregate_mode_expand_columns(tmp_path: Path) -> None:
    """Test expand_columns: AD is expanded into AD_0, AD_1 instead of aggregated."""
    vcf_txt = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=1000000>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">\n'
        '##FORMAT=<ID=RL,Number=.,Type=Integer,Description="Read lengths">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\n"
        "chr1\t100\t.\tA\tG\t30.0\tPASS\t.\tGT:AD:RL\t0/1:10,25:100,101,102\n"
        "chr1\t200\t.\tC\tT\t25.0\tPASS\t.\tGT:AD:RL\t0/1:5,15:200,201\n"
    )
    plain = tmp_path / "expand_columns.vcf"
    plain.write_text(vcf_txt)
    vcf_gz = tmp_path / "expand_columns.vcf.gz"
    subprocess.run(["bcftools", "view", str(plain), "-Oz", "-o", str(vcf_gz), "--write-index=tbi"], check=True)

    out = tmp_path / "expand_columns.parquet"
    featuremap_to_dataframe.vcf_to_parquet(
        str(vcf_gz), str(out), drop_format={"GT"}, list_mode="aggregate", expand_columns={"AD": 2}, jobs=1
    )
    expand_df = pl.read_parquet(out)

    # AD split: AD=10,25 -> AD_0=10, AD_1=25; RL aggregated
    expected = pl.DataFrame(
        {
            "POS": [100, 200],
            "AD_0": [10, 5],
            "AD_1": [25, 15],
            "RL_mean": [101.0, 200.5],
            "RL_count": [3, 2],
        }
    )
    _assert_df_equal(expand_df, expected)

    # AD should NOT have aggregation columns
    assert "AD_mean" not in expand_df.columns and "AD_count" not in expand_df.columns


def test_aggregate_mode_expand_columns_multi_sample(tmp_path: Path) -> None:
    """Test expand_columns with multi-sample VCF: expanded columns get sample suffixes."""
    vcf_txt = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=1000000>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\tSAMPLE2\n"
        "chr1\t100\t.\tA\tG\t30.0\tPASS\t.\tGT:AD\t0/1:10,25\t0/0:35,5\n"
        "chr1\t200\t.\tC\tT\t25.0\tPASS\t.\tGT:AD\t0/1:5,15\t0/0:30,0\n"
    )
    plain = tmp_path / "expand_columns_multi.vcf"
    plain.write_text(vcf_txt)
    vcf_gz = tmp_path / "expand_columns_multi.vcf.gz"
    subprocess.run(["bcftools", "view", str(plain), "-Oz", "-o", str(vcf_gz), "--write-index=tbi"], check=True)

    out = tmp_path / "expand_columns_multi.parquet"
    featuremap_to_dataframe.vcf_to_parquet(
        str(vcf_gz), str(out), drop_format={"GT"}, list_mode="aggregate", expand_columns={"AD": 2}, jobs=1
    )
    multi_expand_df = pl.read_parquet(out)

    # AD split with sample suffixes: SAMPLE1 AD=10,25, SAMPLE2 AD=35,5
    expected = pl.DataFrame(
        {
            "POS": [100, 200],
            "AD_0_SAMPLE1": [10, 5],
            "AD_1_SAMPLE1": [25, 15],
            "AD_0_SAMPLE2": [35, 30],
            "AD_1_SAMPLE2": [5, 0],
        }
    )
    _assert_df_equal(multi_expand_df, expected)
    # AD should NOT have aggregation columns
    assert "AD_mean_SAMPLE1" not in multi_expand_df.columns and "AD_count_SAMPLE1" not in multi_expand_df.columns
    assert "AD_mean_SAMPLE2" not in multi_expand_df.columns and "AD_count_SAMPLE2" not in multi_expand_df.columns


def test_expand_columns_raises_error_in_explode_mode(input_featuremap: Path) -> None:
    """Test that expand_columns raises ValueError when used with explode mode."""

    # Should raise ValueError when expand_columns is used with explode mode (default)
    with pytest.raises(ValueError, match="expand_columns is not supported in explode mode"):
        featuremap_to_dataframe.vcf_to_parquet(
            str(input_featuremap), "output.parquet", drop_format={"GT"}, expand_columns={"AD": 2}, jobs=1
        )

    # Should also raise when explicitly using explode mode
    with pytest.raises(ValueError, match="expand_columns is not supported in explode mode"):
        featuremap_to_dataframe.vcf_to_parquet(
            str(input_featuremap),
            "output.parquet",
            drop_format={"GT"},
            list_mode="explode",
            expand_columns={"AD": 2},
            jobs=1,
        )


def test_expand_columns_rejects_invalid_sizes(input_featuremap: Path) -> None:
    """Test that expand_columns rejects zero or negative sizes."""
    # Test zero size
    with pytest.raises(ValueError, match="expand_columns size must be positive"):
        featuremap_to_dataframe.vcf_to_parquet(
            str(input_featuremap),
            "output.parquet",
            drop_format={"GT"},
            list_mode="aggregate",
            expand_columns={"AD": 0},
            jobs=1,
        )

    # Test negative size
    with pytest.raises(ValueError, match="expand_columns size must be positive"):
        featuremap_to_dataframe.vcf_to_parquet(
            str(input_featuremap),
            "output.parquet",
            drop_format={"GT"},
            list_mode="aggregate",
            expand_columns={"AD": -1},
            jobs=1,
        )


def test_expand_columns_rejects_scalar_format(tmp_path: Path) -> None:
    """Test that expand_columns rejects scalar FORMAT fields."""
    vcf_txt = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=1000000>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
        '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\n"
        "chr1\t100\t.\tA\tG\t30.0\tPASS\t.\tGT:DP:AD\t0/1:12:10,25\n"
    )
    plain = tmp_path / "expand_columns_scalar.vcf"
    plain.write_text(vcf_txt)
    vcf_gz = tmp_path / "expand_columns_scalar.vcf.gz"
    subprocess.run(["bcftools", "view", str(plain), "-Oz", "-o", str(vcf_gz), "--write-index=tbi"], check=True)

    with pytest.raises(ValueError, match="Scalar fields: DP"):
        featuremap_to_dataframe.vcf_to_parquet(
            str(vcf_gz),
            str(tmp_path / "expand_columns_scalar.parquet"),
            drop_format={"GT"},
            list_mode="aggregate",
            expand_columns={"DP": 1},
            jobs=1,
        )
