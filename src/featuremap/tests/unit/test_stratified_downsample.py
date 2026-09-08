"""Unit tests for the indel/SNV-balanced (stratified) downsample in featuremap_to_dataframe.

Covers `_stratified_downsample_by_variant_type` and the guard logic in
`_merge_parquet_files_lazy` that decides between the uniform and stratified paths.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
from ugbio_featuremap.featuremap_to_dataframe import (
    COL_X_IC,
    _merge_parquet_files_lazy,
    _stratified_downsample_by_variant_type,
)

SEED = 12345


def _write_part(path: Path, n_indel: int, n_snv: int, tag: str) -> str:
    """Write a small parquet part-file with n_indel indel rows and n_snv SNV rows.

    Indels alternate ins/del in X_IC; SNVs have a null X_IC. A unique row id and the
    part tag are stored so we can verify no duplication / correct provenance.
    """
    x_ic = ["ins" if i % 2 == 0 else "del" for i in range(n_indel)] + [None] * n_snv
    rid = list(range(n_indel + n_snv))
    result_df = pl.DataFrame(
        {
            "CHROM": ["chr1"] * (n_indel + n_snv),
            "POS": rid,
            COL_X_IC: x_ic,
            "part": [tag] * (n_indel + n_snv),
        }
    )
    result_df.write_parquet(path)
    return str(path)


def _indel_pred() -> pl.Expr:
    return pl.col(COL_X_IC).is_in(["ins", "del"])


# ---------------------------------------------------------------------------
# Helper-level behavior
# ---------------------------------------------------------------------------
def test_realized_fraction_matches_target_when_supply(tmp_path: Path) -> None:
    # Plenty of both pools: 5000 indels, 5000 SNVs, target 40% of 2000 rows.
    parts = [_write_part(tmp_path / f"p{i}.parquet", 2500, 2500, f"p{i}") for i in range(2)]
    out = tmp_path / "out.parquet"
    _stratified_downsample_by_variant_type(parts, str(out), 2000, SEED, 0.4)

    result_df = pl.read_parquet(out)
    assert result_df.height == 2000
    frac = result_df.filter(_indel_pred()).height / result_df.height
    assert frac == pytest.approx(0.4, abs=0.02)
    # Part files cleaned up.
    assert all(not Path(p).exists() for p in parts)


def test_take_all_indels_when_scarce(tmp_path: Path) -> None:
    # Only 50 indels available, but target would want 0.5 * 2000 = 1000.
    parts = [_write_part(tmp_path / "p0.parquet", 50, 10000, "p0")]
    out = tmp_path / "out.parquet"
    _stratified_downsample_by_variant_type(parts, str(out), 2000, SEED, 0.5)

    result_df = pl.read_parquet(out)
    assert result_df.height == 2000
    # All 50 indels kept (never upsampled beyond supply), rest filled with SNVs.
    assert result_df.filter(_indel_pred()).height == 50
    assert result_df.height - result_df.filter(_indel_pred()).height == 1950


def test_snv_scarce_refills_with_indels(tmp_path: Path) -> None:
    # Few SNVs: 100 SNV, 5000 indel, target total 2000, indel target frac 0.4 -> want 800 indel,
    # but SNV supply is only 100, so indels refill to 1900.
    parts = [_write_part(tmp_path / "p0.parquet", 5000, 100, "p0")]
    out = tmp_path / "out.parquet"
    _stratified_downsample_by_variant_type(parts, str(out), 2000, SEED, 0.4)

    result_df = pl.read_parquet(out)
    assert result_df.height == 2000
    assert result_df.height - result_df.filter(_indel_pred()).height == 100
    assert result_df.filter(_indel_pred()).height == 1900


def test_deterministic_same_seed(tmp_path: Path) -> None:
    parts_a = [_write_part(tmp_path / f"a{i}.parquet", 1500, 3000, f"p{i}") for i in range(3)]
    out_a = tmp_path / "a.parquet"
    _stratified_downsample_by_variant_type(parts_a, str(out_a), 2000, SEED, 0.3)

    parts_b = [_write_part(tmp_path / f"b{i}.parquet", 1500, 3000, f"p{i}") for i in range(3)]
    out_b = tmp_path / "b.parquet"
    _stratified_downsample_by_variant_type(parts_b, str(out_b), 2000, SEED, 0.3)

    df_a = pl.read_parquet(out_a)
    df_b = pl.read_parquet(out_b)
    assert df_a.equals(df_b)


# ---------------------------------------------------------------------------
# Guard / dispatch behavior in _merge_parquet_files_lazy
# ---------------------------------------------------------------------------
def test_noop_when_fraction_none_multipart(tmp_path: Path) -> None:
    # fraction None -> uniform path; result is a uniform downsample (no rebalance).
    parts = [_write_part(tmp_path / f"p{i}.parquet", 100, 4900, f"p{i}") for i in range(2)]
    out = tmp_path / "out.parquet"
    _merge_parquet_files_lazy(
        parts, str(out), downsample_reads=2000, downsample_seed=SEED, indel_snv_balance_fraction=None
    )
    result_df = pl.read_parquet(out)
    assert result_df.height == 2000
    # Uniform sampling keeps roughly the original 2% indel ratio, NOT a rebalanced fraction.
    frac = result_df.filter(_indel_pred()).height / result_df.height
    assert frac < 0.1


def test_noop_when_no_x_ic_column(tmp_path: Path) -> None:
    # No X_IC column -> must fall back to uniform path even if fraction is set.
    df_in = pl.DataFrame({"CHROM": ["chr1"] * 5000, "POS": list(range(5000))})
    p = tmp_path / "p0.parquet"
    df_in.write_parquet(p)
    p2 = tmp_path / "p1.parquet"
    df_in.write_parquet(p2)
    out = tmp_path / "out.parquet"
    _merge_parquet_files_lazy(
        [str(p), str(p2)], str(out), downsample_reads=2000, downsample_seed=SEED, indel_snv_balance_fraction=0.4
    )
    result_df = pl.read_parquet(out)
    assert result_df.height == 2000
    assert COL_X_IC not in result_df.columns


def test_noop_when_no_indels(tmp_path: Path) -> None:
    # X_IC column present but all null (no indels) -> uniform path.
    parts = [_write_part(tmp_path / "p0.parquet", 0, 5000, "p0")]
    out = tmp_path / "out.parquet"
    _merge_parquet_files_lazy(
        parts, str(out), downsample_reads=2000, downsample_seed=SEED, indel_snv_balance_fraction=0.4
    )
    result_df = pl.read_parquet(out)
    assert result_df.height == 2000


def test_noop_when_fits_within_downsample(tmp_path: Path) -> None:
    # total rows <= downsample_reads -> keep all (uniform path), no rebalance.
    parts = [_write_part(tmp_path / "p0.parquet", 100, 400, "p0")]
    out = tmp_path / "out.parquet"
    _merge_parquet_files_lazy(
        parts, str(out), downsample_reads=2000, downsample_seed=SEED, indel_snv_balance_fraction=0.4
    )
    result_df = pl.read_parquet(out)
    assert result_df.height == 500  # all kept
    assert result_df.filter(_indel_pred()).height == 100


def test_stratified_via_merge_singlepart(tmp_path: Path) -> None:
    # Single part-file that triggers the stratified path through the public merge entry point.
    parts = [_write_part(tmp_path / "p0.parquet", 3000, 3000, "p0")]
    out = tmp_path / "out.parquet"
    _merge_parquet_files_lazy(
        parts, str(out), downsample_reads=2000, downsample_seed=SEED, indel_snv_balance_fraction=0.4
    )
    result_df = pl.read_parquet(out)
    assert result_df.height == 2000
    frac = result_df.filter(_indel_pred()).height / result_df.height
    assert frac == pytest.approx(0.4, abs=0.02)


def test_stratified_via_merge_multipart(tmp_path: Path) -> None:
    parts = [_write_part(tmp_path / f"p{i}.parquet", 1000, 1000, f"p{i}") for i in range(4)]
    out = tmp_path / "out.parquet"
    _merge_parquet_files_lazy(
        parts, str(out), downsample_reads=3000, downsample_seed=SEED, indel_snv_balance_fraction=0.25
    )
    result_df = pl.read_parquet(out)
    assert result_df.height == 3000
    frac = result_df.filter(_indel_pred()).height / result_df.height
    assert frac == pytest.approx(0.25, abs=0.02)
