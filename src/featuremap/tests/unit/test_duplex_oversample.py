"""Unit tests for duplex oversampling in featuremap_to_dataframe (PE→SE unpaired duplex data).

Covers:
- _add_duplex_group_column: DS-based classification into duplex_pe_full / duplex / ssc / singleton, keyed on
  CS (never MI, which is per-strand on this data). DS==2 = full PE duplex (coverage-based, tags the molecule
  even when only one strand flagged); DS!=2 + both strands flag = duplex; one strand = ssc; no consensus =
  singleton.
- _priority_preserve_downsample: fills the budget top-down by tier (higher tiers kept whole, only the
  boundary tier subsampled).
- _balanced_sample: keeps ~the requested indel fraction within a sampled group.
- _merge_parquet_files_lazy gate: no-op when priority unset or when the CS column is absent.
"""

import polars as pl
from ugbio_featuremap.featuremap_to_dataframe import (
    DUPLEX_GROUP_COL,
    DUPLEX_GROUP_DUPLEX,
    DUPLEX_GROUP_DUPLEX_PE_FULL,
    DUPLEX_GROUP_SINGLETON,
    DUPLEX_GROUP_SSC,
    DUPLEX_PRIORITY_FP,
    _add_duplex_group_column,
    _balanced_sample,
    _merge_parquet_files_lazy,
    _priority_preserve_downsample,
)


def _row(chrom, pos, ref, alt, cs, nf, nr, ds=None, x_ic=None):
    return {
        "CHROM": chrom,
        "POS": pos,
        "REF": ref,
        "ALT": alt,
        "CS": cs,
        "nf": nf,
        "nr": nr,
        "DS": ds,
        "X_IC": x_ic,
    }


def test_duplex_group_classification():
    rows = [
        # duplex_pe_full: DS==2 marks a full PE duplex molecule even though only the fwd strand flags here
        # (the mate strand matched ref and emitted no row) — the cs11291171337-style informative case.
        _row("chr1", 100, "C", "T", "cs_full1", 3, 0, ds=2),
        # duplex_pe_full: DS==2 with both strands flagging
        _row("chr1", 110, "C", "T", "cs_full2", 3, 0, ds=2),
        _row("chr1", 110, "C", "T", "cs_full2", 0, 3, ds=2),
        # duplex: DS not 2 (null) but this CS flags the variant on BOTH strands (chr5-style)
        _row("chr1", 200, "C", "T", "cs_dup", 2, 0, ds=None),
        _row("chr1", 200, "C", "T", "cs_dup", 0, 2, ds=None),
        # ssc: DS not 2, single-strand consensus
        _row("chr1", 300, "C", "T", "cs_ssc", 4, 0, ds=0),
        # singleton: no consensus family
        _row("chr1", 400, "C", "T", "cs_sing", 0, 0, ds=None),
        # duplex_pe_full on an indel (one strand flags) — DS==2 still wins
        _row("chr1", 500, "C", "CA", "cs_full3", 3, 0, ds=2, x_ic="ins"),
        # same CS as cs_dup but a DIFFERENT variant/pos with only one strand -> ssc (the CS+variant key keeps
        # the two loci separate, so it is not promoted to duplex by the other locus).
        _row("chr1", 600, "C", "T", "cs_dup", 5, 0, ds=None),
    ]
    frame = pl.DataFrame(rows)
    g = _add_duplex_group_column(frame)[DUPLEX_GROUP_COL].to_list()
    assert g[0] == DUPLEX_GROUP_DUPLEX_PE_FULL
    assert g[1] == DUPLEX_GROUP_DUPLEX_PE_FULL
    assert g[2] == DUPLEX_GROUP_DUPLEX_PE_FULL
    assert g[3] == DUPLEX_GROUP_DUPLEX
    assert g[4] == DUPLEX_GROUP_DUPLEX
    assert g[5] == DUPLEX_GROUP_SSC
    assert g[6] == DUPLEX_GROUP_SINGLETON
    assert g[7] == DUPLEX_GROUP_DUPLEX_PE_FULL
    assert g[8] == DUPLEX_GROUP_SSC


def test_ds_full_takes_precedence_over_single_strand():
    # A DS==2 read with only one strand flagging is still duplex_pe_full (coverage, not flagging).
    frame = pl.DataFrame([_row("chr1", 100, "C", "T", "m", 7, 0, ds=2)])
    assert _add_duplex_group_column(frame)[DUPLEX_GROUP_COL].to_list() == [DUPLEX_GROUP_DUPLEX_PE_FULL]


def _write_parts(tmp_path, rows_list):
    files = []
    for i, rows in enumerate(rows_list):
        f = str(tmp_path / f"part{i}.parquet")
        pl.DataFrame(rows).write_parquet(f)
        files.append(f)
    return files


def test_priority_preserve_keeps_higher_tiers(tmp_path):
    # 10 duplex_pe_full, 10 duplex (5 molecules x 2 strands), 100 ssc; budget 25.
    rows = []
    for j in range(10):
        rows.append(_row("chr1", 1000 + j, "C", "T", f"f{j}", 3, 0, ds=2))  # duplex_pe_full
    for j in range(5):
        rows += [
            _row("chr1", 2000 + j, "C", "T", f"d{j}", 3, 0, ds=None),
            _row("chr1", 2000 + j, "C", "T", f"d{j}", 0, 3, ds=None),
        ]  # duplex (both strands flag)
    for j in range(100):
        rows.append(_row("chr1", 3000 + j, "C", "T", f"s{j}", 3, 0, ds=None))  # ssc
    files = _write_parts(tmp_path, [rows])
    out = str(tmp_path / "out.parquet")
    _priority_preserve_downsample(
        files,
        out,
        downsample_reads=25,
        downsample_seed=1,
        duplex_oversample_priority=DUPLEX_PRIORITY_FP,
        indel_snv_balance_fraction=None,
    )
    res = _add_duplex_group_column(pl.read_parquet(out))
    counts = dict(res.group_by(DUPLEX_GROUP_COL).len().iter_rows())
    assert res.height == 25
    # full (10) + duplex (10) kept whole; ssc is the boundary (fills remaining 5).
    assert counts.get(DUPLEX_GROUP_DUPLEX_PE_FULL, 0) == 10
    assert counts.get(DUPLEX_GROUP_DUPLEX, 0) == 10
    assert counts.get(DUPLEX_GROUP_SSC, 0) == 5


def test_priority_preserve_drops_ssc_when_duplex_overflows(tmp_path):
    # full (30) alone exceeds budget 20 -> full is the boundary, duplex+ssc dropped; all kept are DS==2.
    rows = [_row("chr1", 1000 + j, "C", "T", f"f{j}", 3, 0, ds=2) for j in range(30)]
    rows += [_row("chr1", 3000 + j, "C", "T", f"s{j}", 3, 0, ds=None) for j in range(50)]  # ssc
    files = _write_parts(tmp_path, [rows])
    out = str(tmp_path / "out.parquet")
    _priority_preserve_downsample(
        files,
        out,
        downsample_reads=20,
        downsample_seed=1,
        duplex_oversample_priority=DUPLEX_PRIORITY_FP,
        indel_snv_balance_fraction=None,
    )
    res = pl.read_parquet(out)
    assert res.height == 20
    assert res.filter(pl.col("DS") == 2).height == 20  # only full-duplex reads survived


def test_balanced_sample_keeps_indel_fraction():
    rows = [_row("chr1", 100 + j, "C", "CA", f"i{j}", 3, 0, ds=2, x_ic="ins") for j in range(20)]
    rows += [_row("chr1", 500 + j, "C", "T", f"s{j}", 3, 0, ds=2) for j in range(80)]
    frame = pl.DataFrame(rows)
    sampled = _balanced_sample(frame, n=20, indel_snv_balance_fraction=0.5, seed=1)
    assert sampled.height == 20
    assert sampled.filter(pl.col("X_IC").is_in(["ins", "del"])).height == 10


def test_merge_gate_noop_without_cs_column(tmp_path):
    # priority set but no CS column -> falls through to the uniform downsample (still trims to budget).
    rows = [{"CHROM": "chr1", "POS": j, "REF": "C", "ALT": "T"} for j in range(100)]
    files = _write_parts(tmp_path, [rows])
    out = str(tmp_path / "out.parquet")
    _merge_parquet_files_lazy(
        files, out, downsample_reads=30, downsample_seed=1, duplex_oversample_priority=DUPLEX_PRIORITY_FP
    )
    res = pl.read_parquet(out)
    assert res.height == 30
    assert DUPLEX_GROUP_COL not in res.columns  # duplex path not taken


def test_merge_gate_noop_when_priority_unset(tmp_path):
    rows = [_row("chr1", j, "C", "T", f"m{j}", 3, 0, ds=2) for j in range(100)]
    files = _write_parts(tmp_path, [rows])
    out = str(tmp_path / "out.parquet")
    _merge_parquet_files_lazy(files, out, downsample_reads=30, downsample_seed=1, duplex_oversample_priority=None)
    res = pl.read_parquet(out)
    assert res.height == 30
    assert DUPLEX_GROUP_COL not in res.columns
