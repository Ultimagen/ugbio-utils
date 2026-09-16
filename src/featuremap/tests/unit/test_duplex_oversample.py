"""Unit tests for duplex/concordance oversampling in featuremap_to_dataframe.

Covers:
- _add_duplex_group_column: classification into concordant_duplex / discordant_duplex / ssc / singleton,
  for SNV and indel rows, using only MI/nf/nr + within-molecule agreement.
- _priority_preserve_downsample: fills the budget top-down by tier (higher tiers kept whole, only the
  boundary tier subsampled), FP 4-tier and TP-style orderings.
- _balanced_sample: keeps ~the requested indel fraction within a sampled group.
- _merge_parquet_files_lazy gate: no-op when priority unset or when the MI column is absent.
"""

import polars as pl
from ugbio_featuremap.featuremap_to_dataframe import (
    DUPLEX_GROUP_COL,
    DUPLEX_GROUP_CONCORDANT,
    DUPLEX_GROUP_DISCORDANT,
    DUPLEX_GROUP_SINGLETON,
    DUPLEX_GROUP_SSC,
    _add_duplex_group_column,
    _balanced_sample,
    _merge_parquet_files_lazy,
    _priority_preserve_downsample,
)

FP_PRIORITY = [DUPLEX_GROUP_CONCORDANT, DUPLEX_GROUP_DISCORDANT, DUPLEX_GROUP_SSC, DUPLEX_GROUP_SINGLETON]


def _row(chrom, pos, ref, alt, mi, nf, nr, x_ic=None):
    return {
        "CHROM": chrom,
        "POS": pos,
        "REF": ref,
        "ALT": alt,
        "MI": mi,
        "nf": nf,
        "nr": nr,
        "X_IC": x_ic,
    }


def test_duplex_group_classification():
    rows = [
        # concordant duplex: same MI, opposite strands, same SNV C->T at chr1:100
        _row("chr1", 100, "C", "T", "m1", 3, 0),
        _row("chr1", 100, "C", "T", "m1", 0, 4),
        # discordant duplex: MI present, only one strand emits the alt here (mate matched ref -> no row)
        _row("chr1", 200, "C", "T", "m2", 5, 0),
        # discordant duplex: same MI+pos but the two strands disagree on the alt (T vs G)
        _row("chr1", 300, "C", "T", "m3", 2, 0),
        _row("chr1", 300, "C", "G", "m3", 0, 2),
        # SSC: consensus single-strand, no MI
        _row("chr1", 400, "C", "T", None, 6, 0),
        # singleton: no consensus family, no MI
        _row("chr1", 500, "C", "T", None, 0, 0),
        # concordant duplex on an indel (ins) — indel descriptors match on both strands
        _row("chr1", 600, "C", "CA", "m4", 3, 0, x_ic="ins"),
        _row("chr1", 600, "C", "CA", "m4", 0, 3, x_ic="ins"),
    ]
    frame = pl.DataFrame(rows)
    out = _add_duplex_group_column(frame)
    g = out[DUPLEX_GROUP_COL].to_list()
    assert g[0] == DUPLEX_GROUP_CONCORDANT
    assert g[1] == DUPLEX_GROUP_CONCORDANT
    assert g[2] == DUPLEX_GROUP_DISCORDANT
    assert g[3] == DUPLEX_GROUP_DISCORDANT
    assert g[4] == DUPLEX_GROUP_DISCORDANT
    assert g[5] == DUPLEX_GROUP_SSC
    assert g[6] == DUPLEX_GROUP_SINGLETON
    assert g[7] == DUPLEX_GROUP_CONCORDANT
    assert g[8] == DUPLEX_GROUP_CONCORDANT


def _write_parts(tmp_path, rows_list):
    files = []
    for i, rows in enumerate(rows_list):
        f = str(tmp_path / f"part{i}.parquet")
        pl.DataFrame(rows).write_parquet(f)
        files.append(f)
    return files


def test_priority_preserve_keeps_higher_tiers(tmp_path):
    # 10 concordant, 10 discordant, 100 ssc, 100 singleton; budget 40.
    rows = []
    for j in range(10):
        rows += [_row("chr1", 1000 + j, "C", "T", f"c{j}", 3, 0), _row("chr1", 1000 + j, "C", "T", f"c{j}", 0, 3)]
    for j in range(10):
        rows.append(_row("chr1", 2000 + j, "C", "T", f"d{j}", 3, 0))  # discordant (lone strand)
    for j in range(100):
        rows.append(_row("chr1", 3000 + j, "C", "T", None, 3, 0))  # ssc
    for j in range(100):
        rows.append(_row("chr1", 4000 + j, "C", "T", None, 0, 0))  # singleton
    files = _write_parts(tmp_path, [rows])
    out = str(tmp_path / "out.parquet")
    _priority_preserve_downsample(
        files,
        out,
        downsample_reads=40,
        downsample_seed=1,
        duplex_oversample_priority=FP_PRIORITY,
        indel_snv_balance_fraction=None,
    )
    res = _add_duplex_group_column(pl.read_parquet(out))
    counts = dict(res.group_by(DUPLEX_GROUP_COL).len().iter_rows())
    assert res.height == 40
    # 20 concordant + 10 discordant kept fully; ssc is the boundary (fills remaining 10); singleton dropped.
    assert counts.get(DUPLEX_GROUP_CONCORDANT, 0) == 20
    assert counts.get(DUPLEX_GROUP_DISCORDANT, 0) == 10
    assert counts.get(DUPLEX_GROUP_SSC, 0) == 10
    assert counts.get(DUPLEX_GROUP_SINGLETON, 0) == 0


def test_priority_preserve_boundary_is_top_tier_when_it_overflows(tmp_path):
    # concordant alone (50) exceeds budget 20 -> concordant is the boundary, everything else dropped.
    rows = []
    for j in range(25):
        rows += [_row("chr1", 1000 + j, "C", "T", f"c{j}", 3, 0), _row("chr1", 1000 + j, "C", "T", f"c{j}", 0, 3)]
    for j in range(50):
        rows.append(_row("chr1", 3000 + j, "C", "T", None, 3, 0))  # ssc
    files = _write_parts(tmp_path, [rows])
    out = str(tmp_path / "out.parquet")
    _priority_preserve_downsample(
        files,
        out,
        downsample_reads=20,
        downsample_seed=1,
        duplex_oversample_priority=FP_PRIORITY,
        indel_snv_balance_fraction=None,
    )
    res = pl.read_parquet(out)
    assert res.height == 20
    # concordant is the boundary tier: subsampling it breaks some molecule pairs, so re-deriving the group
    # on the output is not meaningful. What matters is that all 20 kept reads are duplex-origin (have an MI)
    # and none are SSC (MI-less) — i.e. the highest tier was preserved over SSC.
    assert res.filter(pl.col("MI").is_not_null()).height == 20
    assert res.filter(pl.col("MI").is_null()).height == 0


def test_balanced_sample_keeps_indel_fraction():
    rows = [_row("chr1", 100 + j, "C", "CA", f"i{j}", 3, 0, x_ic="ins") for j in range(20)]
    rows += [_row("chr1", 500 + j, "C", "T", None, 3, 0) for j in range(80)]
    frame = pl.DataFrame(rows)
    sampled = _balanced_sample(frame, n=20, indel_snv_balance_fraction=0.5, seed=1)
    n_indel = sampled.filter(pl.col("X_IC").is_in(["ins", "del"])).height
    assert sampled.height == 20
    assert n_indel == 10  # 0.5 * 20


def test_merge_gate_noop_without_mi_column(tmp_path):
    # priority set but no MI column -> falls through to the uniform downsample (still trims to budget).
    rows = [{"CHROM": "chr1", "POS": j, "REF": "C", "ALT": "T"} for j in range(100)]
    files = _write_parts(tmp_path, [rows])
    out = str(tmp_path / "out.parquet")
    _merge_parquet_files_lazy(
        files, out, downsample_reads=30, downsample_seed=1, duplex_oversample_priority=FP_PRIORITY
    )
    res = pl.read_parquet(out)
    assert res.height == 30
    assert DUPLEX_GROUP_COL not in res.columns  # duplex path not taken


def test_merge_gate_noop_when_priority_unset(tmp_path):
    rows = [_row("chr1", j, "C", "T", "m1" if j % 2 else None, 3, 0) for j in range(100)]
    files = _write_parts(tmp_path, [rows])
    out = str(tmp_path / "out.parquet")
    _merge_parquet_files_lazy(files, out, downsample_reads=30, downsample_seed=1, duplex_oversample_priority=None)
    res = pl.read_parquet(out)
    assert res.height == 30
    assert DUPLEX_GROUP_COL not in res.columns
