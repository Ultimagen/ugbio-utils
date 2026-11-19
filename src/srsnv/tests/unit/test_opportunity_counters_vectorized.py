"""The vectorized hmer-run histogram / STR-locus counters must equal the original per-base Python scans exactly."""

import random

import pysam
import pytest
from ugbio_srsnv.srsnv_training import (
    _seq_codes,
    _str_loci_in_sequence,
    count_str_loci_in_interval_list,
    homopolymer_run_length_histogram,
)


# --- verbatim copies of the original (pre-vectorization) per-base scans, used as references ---------------
def _ref_hmer_hist(seq: str) -> dict[int, int]:
    hist: dict[int, int] = {}
    seq = seq.upper()
    n = len(seq)
    i = 0
    while i < n:
        j = i
        while j < n and seq[j] == seq[i]:
            j += 1
        if seq[i] in "ACGT":
            hist[j - i] = hist.get(j - i, 0) + 1
        i = j
    return hist


def _ref_str_count(seq: str, min_unit=2, max_unit=6, min_copies=2) -> int:
    seq = seq.upper()
    n_str = 0
    n = len(seq)
    i = 0
    while i < n:
        best = 0
        for u in range(min_unit, max_unit + 1):
            if i + min_copies * u > n:
                continue
            unit = seq[i : i + u]
            if len(set(unit)) < 2:  # noqa: PLR2004
                continue
            k = 0
            while i + (k + 1) * u <= n and seq[i + k * u : i + (k + 1) * u] == unit:
                k += 1
            if k >= min_copies and k * u > best:
                best = k * u
        if best:
            n_str += 1
            i += best
        else:
            i += 1
    return n_str


def _random_seq(rng: random.Random, n: int) -> str:
    """Random sequence with planted homopolymers, STRs (units 2..7), Ns and lowercase stretches."""
    out: list[str] = []
    while sum(map(len, out)) < n:
        r = rng.random()
        if r < 0.15:  # noqa: PLR2004
            out.append(rng.choice("ACGT") * rng.randint(2, 25))
        elif r < 0.35:  # noqa: PLR2004
            u = rng.randint(2, 7)
            out.append("".join(rng.choice("ACGT") for _ in range(u)) * rng.randint(1, 8))
        elif r < 0.38:  # noqa: PLR2004
            out.append("N" * rng.randint(1, 30))
        elif r < 0.42:  # noqa: PLR2004
            out.append("".join(rng.choice("acgt") for _ in range(rng.randint(1, 20))))
        else:
            out.append("".join(rng.choice("ACGT") for _ in range(rng.randint(1, 40))))
    return "".join(out)[:n]


@pytest.mark.parametrize("seed", range(40))
def test_str_and_hmer_match_reference_on_random_sequences(seed):
    rng = random.Random(seed)
    seq = _random_seq(rng, rng.randint(0, 3000))
    assert _str_loci_in_sequence(_seq_codes(seq), 2, 6, 2) == _ref_str_count(seq)
    import numpy as np  # noqa: PLC0415
    from ugbio_srsnv.srsnv_training import _ACGT_CODES, _homopolymer_runs  # noqa: PLC0415

    bases, lengths = _homopolymer_runs(_seq_codes(seq))
    lengths = lengths[np.isin(bases, _ACGT_CODES)]
    got = {int(k): int(v) for k, v in enumerate(np.bincount(lengths)) if v} if lengths.size else {}
    assert got == _ref_hmer_hist(seq)


@pytest.mark.parametrize(("min_unit", "max_unit", "min_copies"), [(2, 6, 2), (2, 4, 3), (3, 6, 2), (2, 2, 2)])
def test_str_param_variants_match_reference(min_unit, max_unit, min_copies):
    rng = random.Random(1234)
    for _ in range(15):
        seq = _random_seq(rng, rng.randint(1, 2000))
        assert _str_loci_in_sequence(_seq_codes(seq), min_unit, max_unit, min_copies) == _ref_str_count(
            seq, min_unit, max_unit, min_copies
        )


@pytest.mark.parametrize("seq", ["", "A", "AC", "ACAC", "AAAA", "ACACA", "NNNN", "ACGACGACG", "ATATATGCGCGC"])
def test_edge_sequences(seq):
    assert _str_loci_in_sequence(_seq_codes(seq), 2, 6, 2) == _ref_str_count(seq)


def test_interval_list_counters_match_reference(tmp_path):
    rng = random.Random(7)
    chroms = {"chrA": _random_seq(rng, 20000), "chrB": _random_seq(rng, 8000)}
    fa = tmp_path / "ref.fa"
    with open(fa, "w") as fh:
        for name, seq in chroms.items():
            fh.write(f">{name}\n")
            for k in range(0, len(seq), 60):
                fh.write(seq[k : k + 60] + "\n")
    pysam.faidx(str(fa))
    intervals = [("chrA", 1, 5000), ("chrA", 4000, 19999), ("chrB", 10, 8000), ("chrB", 100, 100)]
    il = tmp_path / "regions.interval_list"
    with open(il, "w") as fh:
        fh.write("@HD\tVN:1.6\n@SQ\tSN:chrA\tLN:20000\n@SQ\tSN:chrB\tLN:8000\n")
        for c, s, e in intervals:
            fh.write(f"{c}\t{s}\t{e}\t+\t.\n")
    ref_hist: dict[int, int] = {}
    ref_str = 0
    for c, s, e in intervals:
        sub = chroms[c][s - 1 : e]
        for k, v in _ref_hmer_hist(sub).items():
            ref_hist[k] = ref_hist.get(k, 0) + v
        ref_str += _ref_str_count(sub)
    assert homopolymer_run_length_histogram(str(il), str(fa)) == ref_hist
    assert count_str_loci_in_interval_list(str(il), str(fa)) == ref_str


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize(("chunk_size", "lookahead"), [(1, 1), (5, 7), (13, 8), (64, 16)])
def test_windowed_counters_match_reference(seed, chunk_size, lookahead):
    """Tiny windows + short lookahead: repeats/runs crossing window ends must be counted exactly once."""
    from ugbio_srsnv.srsnv_training import _count_str_loci_chunked, _homopolymer_hist_chunked  # noqa: PLC0415

    rng = random.Random(seed)
    seq = _random_seq(rng, rng.randint(0, 1500))
    if seed % 3 == 0:  # a repeat much longer than window + lookahead forces the lookahead to grow
        seq = seq[:100] + "ACG" * 200 + seq[100:]
    codes = _seq_codes(seq)
    got_str = _count_str_loci_chunked(lambda s, e: codes[s:e], codes.size, 2, 6, 2, chunk_size, lookahead)
    assert got_str == _ref_str_count(seq)
    counts = _homopolymer_hist_chunked(lambda s, e: codes[s:e], codes.size, chunk_size)
    assert {k: int(v) for k, v in enumerate(counts) if v} == _ref_hmer_hist(seq)


def test_interval_counters_memory_is_bounded_by_window(tmp_path):
    """A single 8 Mb interval must not allocate interval-sized int64 arrays (chr1 needed ~15 GB)."""
    import tracemalloc  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    n = 8_000_000
    rng = np.random.default_rng(0)
    seq = np.frombuffer(b"ACGT", dtype=np.uint8)[rng.integers(0, 4, n)].tobytes()
    fa = tmp_path / "big.fa"
    with open(fa, "wb") as fh:
        fh.write(b">chrBig\n")
        for k in range(0, n, 1_000_000):
            fh.write(seq[k : k + 1_000_000] + b"\n")
    pysam.faidx(str(fa))
    il = tmp_path / "big.interval_list"
    il.write_text(f"@HD\tVN:1.6\n@SQ\tSN:chrBig\tLN:{n}\nchrBig\t1\t{n}\t+\t.\n")
    for fn in (count_str_loci_in_interval_list, homopolymer_run_length_histogram):
        tracemalloc.start()
        fn(str(il), str(fa))
        _cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        # ~85 MB with 1 Mb windows; the whole-interval scan allocated several int64 arrays of n (~0.5 GB here)
        assert peak < 200_000_000, f"{fn.__name__} peak {peak / 1e6:.0f} MB"
