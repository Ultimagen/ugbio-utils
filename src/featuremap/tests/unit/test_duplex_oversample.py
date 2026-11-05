"""Unit tests for duplex oversampling in featuremap_to_dataframe (paired-end pair-dx duplex data).

Covers:
- _add_duplex_group_column: dx-based classification into duplex_pe_full / duplex / ssc / singleton, keyed on
  CS (never MI, which is per-strand on this data). dx==1 = full PE duplex (coverage-based, tags the molecule
  even when only one strand flagged); no dx + both po strands (F1R2 and F2R1) flag = duplex; one strand = ssc;
  no consensus = singleton. Rows without po carry no strand info (nf/nr is never used as the strand).
- _priority_preserve_downsample: fills the budget top-down by tier (higher tiers kept whole, only the
  boundary tier subsampled).
- _balanced_sample: keeps ~the requested indel fraction within a sampled group.
- _merge_parquet_files_lazy gate: no-op when priority unset or when the CS column is absent.
"""

from pathlib import Path

import polars as pl
import pytest
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
from ugbio_featuremap.featuremap_to_dataframe import main as featuremap_to_dataframe_main


def _row(chrom, pos, ref, alt, cs, nf, nr, dx=None, x_ic=None, po=None):
    return {
        "CHROM": chrom,
        "POS": pos,
        "REF": ref,
        "ALT": alt,
        "CS": cs,
        "nf": nf,
        "nr": nr,
        "dx": dx,
        "po": po,
        "X_IC": x_ic,
    }


def test_duplex_group_classification():
    rows = [
        # duplex_pe_full: dx==1 marks a full PE duplex molecule even though only the fwd strand flags here
        # (the mate strand matched ref and emitted no row) — the cs11291171337-style informative case.
        _row("chr1", 100, "C", "T", "cs_full1", 3, 0, dx=1),
        # duplex_pe_full: dx==1 with both strands flagging
        _row("chr1", 110, "C", "T", "cs_full2", 3, 0, dx=1),
        _row("chr1", 110, "C", "T", "cs_full2", 0, 3, dx=1),
        # duplex: no dx but this CS flags the variant on BOTH strands (chr5-style)
        _row("chr1", 200, "C", "T", "cs_dup", 2, 0, dx=None, po="F1R2"),
        _row("chr1", 200, "C", "T", "cs_dup", 0, 2, dx=None, po="F2R1"),
        # ssc: no dx, single-strand consensus
        _row("chr1", 300, "C", "T", "cs_ssc", 4, 0, dx=None),
        # singleton: no consensus family
        _row("chr1", 400, "C", "T", "cs_sing", 0, 0, dx=None),
        # duplex_pe_full on an indel (one strand flags) — dx==1 still wins
        _row("chr1", 500, "C", "CA", "cs_full3", 3, 0, dx=1, x_ic="ins"),
        # same CS as cs_dup but a DIFFERENT variant/pos with only one strand -> ssc (the CS+variant key keeps
        # the two loci separate, so it is not promoted to duplex by the other locus).
        _row("chr1", 600, "C", "T", "cs_dup", 5, 0, dx=None),
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
    # A dx==1 read with only one strand flagging is still duplex_pe_full (coverage, not flagging).
    frame = pl.DataFrame([_row("chr1", 100, "C", "T", "m", 7, 0, dx=1)])
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
        rows.append(_row("chr1", 1000 + j, "C", "T", f"f{j}", 3, 0, dx=1))  # duplex_pe_full
    for j in range(5):
        rows += [
            _row("chr1", 2000 + j, "C", "T", f"d{j}", 3, 0, dx=None, po="F1R2"),
            _row("chr1", 2000 + j, "C", "T", f"d{j}", 0, 3, dx=None, po="F2R1"),
        ]  # duplex (both strands flag)
    for j in range(100):
        rows.append(_row("chr1", 3000 + j, "C", "T", f"s{j}", 3, 0, dx=None))  # ssc
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
    # full (30) alone exceeds budget 20 -> full is the boundary, duplex+ssc dropped; all kept are dx==1.
    rows = [_row("chr1", 1000 + j, "C", "T", f"f{j}", 3, 0, dx=1) for j in range(30)]
    rows += [_row("chr1", 3000 + j, "C", "T", f"s{j}", 3, 0, dx=None) for j in range(50)]  # ssc
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
    assert res.filter(pl.col("dx") == 1).height == 20  # only full-duplex reads survived


def test_priority_preserve_min_group_fraction_floors_ssc(tmp_path):
    # Same overflow scenario as above (full alone > budget), but a 10% per-group floor must rescue SSC:
    # budget 20, floor = round(0.10*20) = 2 -> ssc gets 2 (not 0), full gets the remaining 18.
    rows = [_row("chr1", 1000 + j, "C", "T", f"f{j}", 3, 0, dx=1) for j in range(90)]
    rows += [_row("chr1", 3000 + j, "C", "T", f"s{j}", 3, 0, dx=None) for j in range(50)]  # ssc
    files = _write_parts(tmp_path, [rows])
    out = str(tmp_path / "out.parquet")
    _priority_preserve_downsample(
        files,
        out,
        downsample_reads=20,
        downsample_seed=1,
        duplex_oversample_priority=DUPLEX_PRIORITY_FP,
        indel_snv_balance_fraction=None,
        min_group_fraction=0.10,
    )
    res = pl.read_parquet(out)
    assert res.height == 20
    full = res.filter(pl.col("dx") == 1).height
    ssc = res.height - full
    assert ssc == 2  # floor honored (was 0 without the floor)
    assert full == 18  # top tier takes only the surplus after the floor


def test_balanced_sample_keeps_indel_fraction():
    rows = [_row("chr1", 100 + j, "C", "CA", f"i{j}", 3, 0, dx=1, x_ic="ins") for j in range(20)]
    rows += [_row("chr1", 500 + j, "C", "T", f"s{j}", 3, 0, dx=1) for j in range(80)]
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
    rows = [_row("chr1", j, "C", "T", f"m{j}", 3, 0, dx=1) for j in range(100)]
    files = _write_parts(tmp_path, [rows])
    out = str(tmp_path / "out.parquet")
    _merge_parquet_files_lazy(files, out, downsample_reads=30, downsample_seed=1, duplex_oversample_priority=None)
    res = pl.read_parquet(out)
    assert res.height == 30
    assert DUPLEX_GROUP_COL not in res.columns


def test_paired_po_strand_not_nf_nr():
    """Paired consensus data: the two mates of ONE strand have opposite nf/nr (alignment orientation) but the same
    po -> single-strand (ssc), not duplex. Both po orientations flagging (no dx) -> duplex; dx==1 -> duplex_pe_full."""
    rows = [
        # simplex pair: R1 (nr>0) + R2 (nf>0), same strand F2R1 -> ssc (nf/nr alone would wrongly say duplex)
        _row("chr1", 100, "C", "T", "cs_simplex", 0, 2, dx=None, po="F2R1"),
        _row("chr1", 100, "C", "T", "cs_simplex", 2, 0, dx=None, po="F2R1"),
        # both strands (F1R2 + F2R1) flag the variant, no dx -> duplex
        _row("chr1", 200, "C", "T", "cs_both", 2, 0, dx=None, po="F1R2"),
        _row("chr1", 200, "C", "T", "cs_both", 0, 2, dx=None, po="F2R1"),
        # dx marks the molecule duplex even with one strand flagging
        _row("chr1", 300, "C", "T", "cs_dx", 2, 0, dx=1, po="F1R2"),
    ]
    g = _add_duplex_group_column(pl.DataFrame(rows))[DUPLEX_GROUP_COL].to_list()
    assert g == [
        DUPLEX_GROUP_SSC,
        DUPLEX_GROUP_SSC,
        DUPLEX_GROUP_DUPLEX,
        DUPLEX_GROUP_DUPLEX,
        DUPLEX_GROUP_DUPLEX_PE_FULL,
    ]


def test_missing_po_is_no_strand_info():
    """Without po (column absent or null) nf/nr is NOT used as the strand: opposite nf/nr rows of one CS stay ssc;
    dx still marks duplex_pe_full and nf==nr==0 is still a singleton."""
    rows = [
        {"CHROM": "chr1", "POS": 100, "REF": "C", "ALT": "T", "CS": "m", "nf": 2, "nr": 0, "dx": None},
        {"CHROM": "chr1", "POS": 100, "REF": "C", "ALT": "T", "CS": "m", "nf": 0, "nr": 2, "dx": None},
        {"CHROM": "chr1", "POS": 200, "REF": "C", "ALT": "T", "CS": "f", "nf": 2, "nr": 0, "dx": 1},
        {"CHROM": "chr1", "POS": 300, "REF": "C", "ALT": "T", "CS": "s", "nf": 0, "nr": 0, "dx": None},
    ]
    expected = [DUPLEX_GROUP_SSC, DUPLEX_GROUP_SSC, DUPLEX_GROUP_DUPLEX_PE_FULL, DUPLEX_GROUP_SINGLETON]
    no_po_col = pl.DataFrame(rows)
    assert _add_duplex_group_column(no_po_col)[DUPLEX_GROUP_COL].to_list() == expected
    null_po = no_po_col.with_columns(pl.lit(None, dtype=pl.Utf8).alias("po"))
    assert _add_duplex_group_column(null_po)[DUPLEX_GROUP_COL].to_list() == expected


def test_ds_is_ignored():
    """DS (family size on the new CRAMs) must not drive the duplex classification."""
    frame = pl.DataFrame([{**_row("chr1", 100, "C", "T", "m", 3, 0, dx=None, po="F1R2"), "DS": 2}])
    assert _add_duplex_group_column(frame)[DUPLEX_GROUP_COL].to_list() == [DUPLEX_GROUP_SSC]


def test_priority_preserve_partial_group_honors_balance(tmp_path):
    # Regression: the plain tier ladder pre-sampled a partial group at ONE uniform rate (~10% here), so the rare
    # indel stratum was decimated before the balanced exact-trim and the realized indel count stayed near its
    # natural 2% (~40 of 400). One ssc tier (the boundary group): 400 indel + 19600 SNV rows across 2 parts;
    # budget 1000, balance 0.5 -> want 500 indels but only 400 exist -> all 400 indels kept + 600 SNVs.
    def _rows(start, n, x_ic):
        return [_row("chr1", start + j, "C", "CA" if x_ic else "T", f"m{start + j}", 3, 0, x_ic=x_ic) for j in range(n)]

    files = _write_parts(
        tmp_path,
        [
            _rows(0, 200, "ins") + _rows(1_000, 9_800, None),
            _rows(100_000, 200, "del") + _rows(200_000, 9_800, None),
        ],
    )
    out = str(tmp_path / "out.parquet")
    _priority_preserve_downsample(
        files,
        out,
        downsample_reads=1000,
        downsample_seed=1,
        duplex_oversample_priority=DUPLEX_PRIORITY_FP,
        indel_snv_balance_fraction=0.5,
    )
    res = pl.read_parquet(out)
    assert res.height == 1000
    assert res.filter(pl.col("X_IC").is_in(["ins", "del"])).height == 400


@pytest.mark.parametrize(
    "priority",
    [["cs_alt_count"], [*DUPLEX_PRIORITY_FP, "cs_alt_count"], [DUPLEX_GROUP_SSC, "bogus"]],
)
def test_unknown_priority_tier_is_rejected(tmp_path, priority):
    # The removed cs_alt_count mode (and any other non-tier value) must fail loudly, naming the valid tiers, rather
    # than being silently ignored by the tier ladder.
    rows = [_row("chr1", j, "C", "T", f"m{j}", 3, 0) for j in range(50)]
    files = _write_parts(tmp_path, [rows])
    out = str(tmp_path / "out.parquet")
    with pytest.raises(ValueError, match="duplex_pe_full, duplex, ssc, singleton") as exc:
        _merge_parquet_files_lazy(
            files, out, downsample_reads=10, downsample_seed=1, duplex_oversample_priority=priority
        )
    assert priority[-1] in str(exc.value)
    assert not Path(out).exists()


def test_unknown_priority_tier_rejected_by_cli(capsys):
    with pytest.raises(SystemExit) as exc:
        featuremap_to_dataframe_main(
            ["--input", "in.vcf.gz", "--output", "out.parquet", "--duplex-oversample-priority", "duplex,cs_alt_count"]
        )
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "cs_alt_count" in err
    assert "duplex_pe_full, duplex, ssc, singleton" in err


def test_valid_priority_tiers_accepted(tmp_path):
    rows = [_row("chr1", j, "C", "T", f"m{j}", 3, 0) for j in range(50)]
    files = _write_parts(tmp_path, [rows])
    out = str(tmp_path / "out.parquet")
    _merge_parquet_files_lazy(
        files, out, downsample_reads=10, downsample_seed=1, duplex_oversample_priority=list(DUPLEX_PRIORITY_FP)
    )
    assert pl.read_parquet(out).height == 10
