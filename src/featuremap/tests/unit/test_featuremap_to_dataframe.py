from __future__ import annotations

import gzip
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
    """Multi-sample VCF should produce columns with sample name prefixes for FORMAT fields."""
    vcf_txt = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr19,length=58617616>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=VAF,Number=1,Type=Float,Description="Variant Allele Frequency">\n'
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Number of reads containing this location">\n'
        '##FORMAT=<ID=RN,Number=.,Type=String,Description="Query (read) name">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tPa_46_FreshFrozen\tPa_46_Buffycoat\n"
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
    # Check that FORMAT fields are prefixed with sample names
    assert "Pa_46_FreshFrozen_VAF" in featuremap_dataframe.columns, "Should have Pa_46_FreshFrozen_VAF column"
    assert "Pa_46_Buffycoat_VAF" in featuremap_dataframe.columns, "Should have Pa_46_Buffycoat_VAF column"
    assert "Pa_46_FreshFrozen_DP" in featuremap_dataframe.columns, "Should have Pa_46_FreshFrozen_DP column"
    assert "Pa_46_Buffycoat_DP" in featuremap_dataframe.columns, "Should have Pa_46_Buffycoat_DP column"
    assert "Pa_46_FreshFrozen_RN" in featuremap_dataframe.columns, "Should have Pa_46_FreshFrozen_RN column"
    assert "Pa_46_Buffycoat_RN" in featuremap_dataframe.columns, "Should have Pa_46_Buffycoat_RN column"
    # Fixed columns should not be prefixed
    assert "CHROM" in featuremap_dataframe.columns, "CHROM should not be prefixed"
    assert "POS" in featuremap_dataframe.columns, "POS should not be prefixed"
    assert "QUAL" in featuremap_dataframe.columns, "QUAL should not be prefixed"
    # Check that data is correctly joined - both samples should have data for the same positions
    assert featuremap_dataframe["POS"].n_unique() == 2, "Should have 2 unique positions"
    # Check that VAF values are correctly preserved
    freshfrozen_vaf = featuremap_dataframe["Pa_46_FreshFrozen_VAF"].to_list()
    buffycoat_vaf = featuremap_dataframe["Pa_46_Buffycoat_VAF"].to_list()
    # At least one row should have the expected VAF values
    assert any(
        v == 0.0298507 for v in freshfrozen_vaf if v is not None
    ), "Should have VAF=0.0298507 for Pa_46_FreshFrozen"
    assert any(v == 0.0 for v in buffycoat_vaf if v is not None), "Should have VAF=0.0 for Pa_46_Buffycoat"


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
    Test that aggregate mode replaces list columns with aggregation metrics (mean, min, max, count).

    This test verifies:
    1. List columns are replaced with 4 columns (mean, min, max, count)
    2. Row count is one per variant (not exploded)
    3. Aggregate columns have correct types (Float64 for mean/min/max, Int64 for count)
    4. Aggregate values are computed correctly
    """
    # Create a VCF with list fields containing numeric values
    vcf_txt = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=1000000>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
        '##FORMAT=<ID=AD,Number=.,Type=Integer,Description="Allelic depths">\n'
        '##FORMAT=<ID=RL,Number=.,Type=Integer,Description="Read lengths">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\n"
        "chr1\t100\t.\tA\tG\t30.0\tPASS\t.\tGT:DP:AD:RL\t0/1:50:10,20,15:100,101,102\n"
        "chr1\t200\t.\tC\tT\t25.0\tPASS\t.\tGT:DP:AD:RL\t0/1:30:5,10:200,201\n"
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

    df_aggregate = pl.read_parquet(out_aggregate)

    # Should have 3 rows (one per variant, not exploded)
    assert df_aggregate.height == 3, "Aggregate mode should have one row per variant"

    # Check that list columns are replaced with aggregate columns
    assert "AD" not in df_aggregate.columns, "Original AD column should be removed"
    assert "RL" not in df_aggregate.columns, "Original RL column should be removed"
    assert "AD_mean" in df_aggregate.columns, "Should have AD_mean column"
    assert "AD_min" in df_aggregate.columns, "Should have AD_min column"
    assert "AD_max" in df_aggregate.columns, "Should have AD_max column"
    assert "AD_count" in df_aggregate.columns, "Should have AD_count column"
    assert "RL_mean" in df_aggregate.columns, "Should have RL_mean column"
    assert "RL_min" in df_aggregate.columns, "Should have RL_min column"
    assert "RL_max" in df_aggregate.columns, "Should have RL_max column"
    assert "RL_count" in df_aggregate.columns, "Should have RL_count column"

    # Check column types
    assert df_aggregate["AD_mean"].dtype == pl.Float64, "AD_mean should be Float64"
    assert df_aggregate["AD_min"].dtype == pl.Float64, "AD_min should be Float64"
    assert df_aggregate["AD_max"].dtype == pl.Float64, "AD_max should be Float64"
    assert df_aggregate["AD_count"].dtype == pl.Int64, "AD_count should be Int64"
    assert df_aggregate["RL_mean"].dtype == pl.Float64, "RL_mean should be Float64"
    assert df_aggregate["RL_min"].dtype == pl.Float64, "RL_min should be Float64"
    assert df_aggregate["RL_max"].dtype == pl.Float64, "RL_max should be Float64"
    assert df_aggregate["RL_count"].dtype == pl.Int64, "RL_count should be Int64"

    # Check aggregate values for first variant (AD: 10,20,15)
    first_row = df_aggregate.filter(pl.col("POS") == 100)
    assert first_row.height == 1, "Should have one row for POS=100"
    ad_mean_val = first_row["AD_mean"].item()
    ad_min_val = first_row["AD_min"].item()
    ad_max_val = first_row["AD_max"].item()
    ad_count_val = first_row["AD_count"].item()

    # AD values: 10, 20, 15 -> mean=15.0, min=10, max=20, count=3
    assert abs(ad_mean_val - 15.0) < 0.0001, f"AD_mean should be 15.0, got {ad_mean_val}"
    assert ad_min_val == 10.0, f"AD_min should be 10.0, got {ad_min_val}"
    assert ad_max_val == 20.0, f"AD_max should be 20.0, got {ad_max_val}"
    assert ad_count_val == 3, f"AD_count should be 3, got {ad_count_val}"

    # Check RL values for first variant (RL: 100,101,102)
    rl_mean_val = first_row["RL_mean"].item()
    rl_min_val = first_row["RL_min"].item()
    rl_max_val = first_row["RL_max"].item()
    rl_count_val = first_row["RL_count"].item()

    # RL values: 100, 101, 102 -> mean=101.0, min=100, max=102, count=3
    assert abs(rl_mean_val - 101.0) < 0.0001, f"RL_mean should be 101.0, got {rl_mean_val}"
    assert rl_min_val == 100.0, f"RL_min should be 100.0, got {rl_min_val}"
    assert rl_max_val == 102.0, f"RL_max should be 102.0, got {rl_max_val}"
    assert rl_count_val == 3, f"RL_count should be 3, got {rl_count_val}"

    # Check second variant (AD: 5,10 -> mean=7.5, min=5, max=10, count=2)
    second_row = df_aggregate.filter(pl.col("POS") == 200)
    assert second_row.height == 1, "Should have one row for POS=200"
    ad_mean_val2 = second_row["AD_mean"].item()
    ad_min_val2 = second_row["AD_min"].item()
    ad_max_val2 = second_row["AD_max"].item()
    ad_count_val2 = second_row["AD_count"].item()

    assert abs(ad_mean_val2 - 7.5) < 0.0001, f"AD_mean should be 7.5, got {ad_mean_val2}"
    assert ad_min_val2 == 5.0, f"AD_min should be 5.0, got {ad_min_val2}"
    assert ad_max_val2 == 10.0, f"AD_max should be 10.0, got {ad_max_val2}"
    assert ad_count_val2 == 2, f"AD_count should be 2, got {ad_count_val2}"

    # Check third variant (AD: . -> mean=., min=., max=., count=0)
    third_row = df_aggregate.filter(pl.col("POS") == 300)
    assert third_row.height == 1, "Should have one row for POS=300"
    ad_mean_val3 = third_row["AD_mean"].item()
    ad_count_val3 = third_row["AD_count"].item()

    # When no valid values, should return "." for mean/min/max and 0 for count
    assert ad_mean_val3 is None or str(ad_mean_val3) == ".", f"AD_mean should be None or '.', got {ad_mean_val3}"
    assert ad_count_val3 == 0, f"AD_count should be 0 for missing values, got {ad_count_val3}"

    # Compare with explode mode to verify row count difference
    out_explode = tmp_path / "explode.parquet"
    featuremap_to_dataframe.vcf_to_parquet(
        str(vcf_gz), str(out_explode), drop_format={"GT"}, list_mode="explode", jobs=1
    )

    df_explode = pl.read_parquet(out_explode)
    # Explode mode should have more rows (one per list element)
    assert df_explode.height > df_aggregate.height, "Explode mode should have more rows than aggregate mode"
