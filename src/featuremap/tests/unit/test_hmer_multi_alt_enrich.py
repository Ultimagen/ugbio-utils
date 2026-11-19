"""--hmer-multi-alt-min-share: raise the share of hmer-indel rows whose CS molecule has >= 2 ALT rows."""

import polars as pl
import pytest
from ugbio_featuremap.featuremap_to_dataframe import (
    DUPLEX_PRIORITY_FP,
    HMER_MULTI_ALT_COL,
    SAMPLE_WEIGHT_COL,
    _add_hmer_multi_alt_column,
    _merge_parquet_files_lazy,
    _priority_preserve_downsample,
)

HM = pl.col("X_INDEL_CLASS") == "hmer"


def _row(rn, pos, cs, x_ic=None, cls=None, ref="C", alt="T"):
    return {
        "RN": rn,
        "CHROM": "chr1",
        "POS": pos,
        "REF": ref,
        "ALT": alt,
        "CS": cs,
        "nf": 2,
        "nr": 0,
        "dx": 1,
        "po": "F1R2",
        "X_IC": x_ic,
        "X_IL": 1 if x_ic else None,
        "X_INDEL_CLASS": cls,
    }


def _pool(n_single=900, n_multi_mol=150, n_snv=1000, n_nonhmer_multi=20):
    rows = [_row(f"s{j}", 10 * j, f"s{j}", "del", "hmer", "CAA", "CA") for j in range(n_single)]
    for j in range(n_multi_mol):  # 2 ALT rows (reads) per molecule
        rows += [_row(f"m{j}_{k}", 100000 + 10 * j, f"m{j}", "del", "hmer", "CAA", "CA") for k in range(2)]
    for j in range(n_nonhmer_multi):  # multi-ALT but non-hmer -> never flagged
        rows += [_row(f"n{j}_{k}", 200000 + 10 * j, f"n{j}", "ins", "non_hmer", "C", "CAG") for k in range(2)]
    rows += [_row(f"v{j}", 300000 + 10 * j, f"v{j}") for j in range(n_snv)]
    return pl.DataFrame(rows)


@pytest.fixture
def pool_flags():
    return _add_hmer_multi_alt_column(_pool()).select("RN", HMER_MULTI_ALT_COL)


def test_flag_only_multi_alt_hmer(pool_flags):
    f = _add_hmer_multi_alt_column(_pool())
    assert f.filter(pl.col(HMER_MULTI_ALT_COL)).height == 300
    assert f.filter(pl.col(HMER_MULTI_ALT_COL) & ~HM).height == 0


def _run(tmp_path, share, n_parts=2):
    # parts split by molecule (molecules never span parts)
    pool = _pool().with_columns((pl.col("POS") // 1000 % n_parts).alias("_p"))
    files = []
    for i in range(n_parts):
        f = str(tmp_path / f"p{i}.parquet")
        pool.filter(pl.col("_p") == i).drop("_p").write_parquet(f)
        files.append(f)
    out = str(tmp_path / f"out_{share}.parquet")
    _priority_preserve_downsample(files, out, 1000, 3, list(DUPLEX_PRIORITY_FP), 0.5, hmer_multi_alt_min_share=share)
    return pl.read_parquet(out)


def _multi_share(df, pool_flags):
    return df.filter(HM).join(pool_flags, on="RN")[HMER_MULTI_ALT_COL].mean()


def test_enrichment_raises_share_keeps_hmer_count(tmp_path, pool_flags):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    base = _run(tmp_path / "a", None)
    enr = _run(tmp_path / "b", 0.3)
    assert HMER_MULTI_ALT_COL not in enr.columns
    assert "_row_id" not in enr.columns
    assert enr.height == base.height == 1000
    assert enr.filter(HM).height == base.filter(HM).height
    assert enr["RN"].n_unique() == enr.height  # no duplicated rows
    assert _multi_share(base, pool_flags) < 0.25
    assert _multi_share(enr, pool_flags) == pytest.approx(0.3, abs=0.01)


def test_enrichment_never_lowers(tmp_path, pool_flags):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    base = _run(tmp_path / "a", None)
    enr = _run(tmp_path / "b", 0.01)
    assert enr.sort("RN").equals(base.sort("RN"))


def test_enrichment_limited_by_pool(tmp_path, pool_flags):
    enr = _run(tmp_path, 0.99)
    assert enr.filter(HM).join(pool_flags, on="RN")[HMER_MULTI_ALT_COL].sum() == 300  # all multi rows taken


def test_sample_weight_conditions_on_flag(tmp_path, pool_flags):
    f = str(tmp_path / "p0.parquet")
    _pool().write_parquet(f)
    out = str(tmp_path / "o.parquet")
    _merge_parquet_files_lazy(
        [f],
        out,
        downsample_reads=1000,
        downsample_seed=1,
        indel_snv_balance_fraction=0.5,
        duplex_oversample_priority=list(DUPLEX_PRIORITY_FP),
        hmer_multi_alt_min_share=0.3,
    )
    o = pl.read_parquet(out).join(pool_flags, on="RN")
    w_multi = o.filter(HM & pl.col(HMER_MULTI_ALT_COL))[SAMPLE_WEIGHT_COL].mean()
    w_single = o.filter(HM & ~pl.col(HMER_MULTI_ALT_COL))[SAMPLE_WEIGHT_COL].mean()
    assert w_multi < w_single  # over-represented multi-ALT rows get the smaller importance weight
