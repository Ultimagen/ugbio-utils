"""Sampling weights emitted by the training-set downsample (featuremap_to_dataframe._merge_parquet_files_lazy).

The downsample oversamples on purpose (duplex priority ladder + floor, indel/SNV balance, cs_alt_count). Each kept
row gets ``sample_weight = n_pre / n_post`` of its sampling stratum, so recalibration can undo the oversampling.
Invariant tested on every path: for every stratum — and therefore for any coarser grouping such as the variant
cell or the DS group — the weights of the kept rows sum exactly to the pre-downsample row count.
"""

import json

import numpy as np
import polars as pl
import pytest
from ugbio_featuremap.featuremap_to_dataframe import (
    DUPLEX_PRIORITY_CS_ALT_COUNT,
    DUPLEX_PRIORITY_FP,
    SAMPLE_WEIGHT_COL,
    _annotate_sampling_strata,
    _merge_parquet_files_dispatch,
    _merge_parquet_files_lazy,
)
from ugbio_featuremap.featuremap_utils import cell_key_expr


def _frame(n_mol=3000, seed=0):
    """Molecules with 1-3 reads; mix of DS==2 / not, SNV / hmer (several run lengths) / non-hmer indels."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_mol):
        kind = rng.choice(["snv", "hmer", "non_hmer"], p=[0.6, 0.3, 0.1])
        ds = 2 if rng.random() < 0.3 else int(rng.choice([0, 1]))
        run = int(rng.choice([1, 2, 3, 5, 8, 12]))
        for r in range(int(rng.integers(1, 4))):
            rows.append(
                {
                    "CHROM": "chr1",
                    "POS": 1000 + m,
                    "REF": "C",
                    "ALT": "T" if kind == "snv" else "CA",
                    "CS": f"cs{m}",
                    "RN": f"r{m}_{r}",
                    "nf": int(rng.integers(0, 3)),
                    "nr": int(rng.integers(0, 3)),
                    "DS": ds,
                    "X_IC": None if kind == "snv" else "ins",
                    "X_INDEL_CLASS": None if kind == "snv" else kind,
                    "X_HMER_RUN": run if kind == "hmer" else None,
                }
            )
    return pl.DataFrame(rows)


def _write_parts(tmp_path, frame, n_parts=3):
    # molecules never span parts: split by molecule (POS)
    files = []
    for i in range(n_parts):
        f = str(tmp_path / f"part{i}.parquet")
        frame.filter(pl.col("POS") % n_parts == i).write_parquet(f)
        files.append(f)
    return files


def _assert_weights_restore(pre: pl.DataFrame, out: pl.DataFrame, by: str, rtol: float = 1e-5):
    pre_n = pre.group_by(by).len().rename({"len": "n_pre"})
    post_w = out.group_by(by).agg(pl.col(SAMPLE_WEIGHT_COL).cast(pl.Float64).sum().alias("w"))
    j = pre_n.join(post_w, on=by, how="inner")
    assert j.height > 0
    np.testing.assert_allclose(j["w"].to_numpy(), j["n_pre"].to_numpy(), rtol=rtol)


def _with_keys(frame):
    return frame.with_columns(
        cell_key_expr(frame.columns).alias("_cell"),
        (pl.col("DS") == 2).alias("_dup"),
    )


def _stratum_labels(tmp_path, frame, **kwargs):
    """Per-RN pre-downsample stratum label, computed exactly as the pipeline does."""
    (tmp_path / "lbl").mkdir()
    files = _write_parts(tmp_path / "lbl", frame)
    prio = kwargs.get("duplex_oversample_priority") or []
    _annotate_sampling_strata(files, use_duplex_group=bool(prio), use_count=DUPLEX_PRIORITY_CS_ALT_COUNT in prio)
    return pl.concat([pl.read_parquet(f, columns=["RN", "_sampling_stratum"]) for f in files])


CASES = {
    "priority": {"duplex_oversample_priority": list(DUPLEX_PRIORITY_FP), "duplex_group_min_fraction": 0.1},
    "priority+balance": {
        "duplex_oversample_priority": list(DUPLEX_PRIORITY_FP),
        "duplex_group_min_fraction": 0.1,
        "indel_snv_balance_fraction": 0.5,
    },
    "priority+cs_alt_count": {
        "duplex_oversample_priority": [*DUPLEX_PRIORITY_FP, DUPLEX_PRIORITY_CS_ALT_COUNT],
        "duplex_group_min_fraction": 0.1,
    },
    "cs_alt_count": {"duplex_oversample_priority": [DUPLEX_PRIORITY_CS_ALT_COUNT], "duplex_group_min_fraction": 0.1},
    "stratified": {"indel_snv_balance_fraction": 0.5},
    "uniform": {},
}


@pytest.mark.parametrize("case", list(CASES))
def test_weights_restore_pre_downsample_counts(tmp_path, case):
    kwargs = CASES[case]
    frame = _frame()
    files = _write_parts(tmp_path, frame)
    out = str(tmp_path / "out.parquet")
    _merge_parquet_files_lazy(files, out, downsample_reads=3000, downsample_seed=3, **kwargs)
    res = pl.read_parquet(out)
    assert res.height <= 3000
    assert SAMPLE_WEIGHT_COL in res.columns
    assert "_sampling_stratum" not in res.columns
    assert res[SAMPLE_WEIGHT_COL].is_not_null().all()
    assert (res[SAMPLE_WEIGHT_COL] >= 1.0 - 1e-6).all()  # downsampling never duplicates rows

    # Exact invariant: for every stratum the sampler kept, weights sum to its pre-downsample count.
    labels = _stratum_labels(tmp_path, frame, **kwargs)
    pre_s = labels.group_by("_sampling_stratum").len()
    post_s = (
        res.join(labels, on="RN").group_by("_sampling_stratum").agg(pl.col(SAMPLE_WEIGHT_COL).cast(pl.Float64).sum())
    )
    j = pre_s.join(post_s, on="_sampling_stratum", how="inner")
    np.testing.assert_allclose(j[SAMPLE_WEIGHT_COL].to_numpy(), j["len"].to_numpy(), rtol=1e-5)
    lost = pre_s.height - j.height

    pre, post = _with_keys(frame), _with_keys(res)
    if lost == 0:
        _assert_weights_restore(pre, post, "_cell")  # cell is part of every stratum key
        assert np.isclose(res[SAMPLE_WEIGHT_COL].cast(pl.Float64).sum(), frame.height, rtol=1e-5)
    else:
        assert "cs_alt_count" in case  # only the count modes drop whole strata here (deterministic by count)
    if "priority" in case and lost == 0:
        _assert_weights_restore(pre, post, "_dup")  # DS==2 <=> duplex_pe_full, a stratum component
    elif not kwargs.get("duplex_oversample_priority"):
        # DS is not a sampling key on these paths: weighted DS share is only statistically restored.
        _assert_weights_restore(pre, post, "_dup", rtol=0.08)


def test_no_weight_column_when_nothing_dropped(tmp_path):
    frame = _frame(n_mol=50)
    files = _write_parts(tmp_path, frame)
    out = str(tmp_path / "out.parquet")
    _merge_parquet_files_lazy(
        files, out, downsample_reads=10_000, downsample_seed=3, duplex_oversample_priority=list(DUPLEX_PRIORITY_FP)
    )
    res = pl.read_parquet(out)
    assert res.height == frame.height
    assert SAMPLE_WEIGHT_COL not in res.columns
    assert "_sampling_stratum" not in res.columns


def test_selection_unchanged_by_stratum_tag(tmp_path):
    """The kept rows must be exactly the rows the sampler kept before weights existed (tag is column-only)."""
    frame = _frame()
    out_a = str(tmp_path / "a.parquet")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    kw = {"duplex_oversample_priority": list(DUPLEX_PRIORITY_FP), "duplex_group_min_fraction": 0.1}
    _merge_parquet_files_dispatch(_write_parts(tmp_path / "a", frame), out_a, 3000, 3, None, **kw)
    out_b = str(tmp_path / "b.parquet")
    _merge_parquet_files_lazy(
        _write_parts(tmp_path / "b", frame), out_b, downsample_reads=3000, downsample_seed=3, **kw
    )
    a = pl.read_parquet(out_a)["RN"].sort().to_list()
    b = pl.read_parquet(out_b)["RN"].sort().to_list()
    assert a == b


def test_per_class_counts_include_hmer_bins(tmp_path):
    frame = _frame()
    files = _write_parts(tmp_path, frame)
    out = str(tmp_path / "out.parquet")
    counts_json = str(tmp_path / "counts.json")
    _merge_parquet_files_lazy(files, out, per_class_counts_json=counts_json)
    counts = json.loads(open(counts_json).read())
    bins = {k: v for k, v in counts.items() if k.startswith("hmer_indel|")}
    assert bins, counts
    assert sum(bins.values()) == counts["hmer_indel"]
    assert {"snv", "hmer_indel", "non_hmer_indel", "total"} <= set(counts)


def test_dropped_stratum_is_reported(tmp_path, caplog):
    """Strict priority without a floor drops whole tiers: their mass can't be re-weighted -> warning."""
    frame = _frame()
    files = _write_parts(tmp_path, frame)
    out = str(tmp_path / "out.parquet")
    with caplog.at_level("WARNING"):
        _merge_parquet_files_lazy(
            files, out, downsample_reads=500, downsample_seed=3, duplex_oversample_priority=list(DUPLEX_PRIORITY_FP)
        )
    res = pl.read_parquet(out)
    assert SAMPLE_WEIGHT_COL in res.columns
    assert res[SAMPLE_WEIGHT_COL].cast(pl.Float64).sum() < frame.height  # lost mass
    assert any("dropped" in r.getMessage() and "sampling strata" in r.getMessage() for r in caplog.records)
