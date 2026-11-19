"""Multi-call (ppmSeq+ dual flow) support in featuremap_to_dataframe.

Every molecule is called twice (R1 / R2 flow orders) and stored as a mate pair sharing qname and strand, so
snvfind --emit-mate tags each read row with MATE (0 unpaired / 1 R1 / 2 R2). Covers:
- the MATE FORMAT field is carried into the parquet (and is absent / harmless for old VCFs);
- the training downsample keeps both calls of a molecule (same CHROM/POS/RN/variant) together, also with the
  indel/SNV balance and when duplex priority is requested;
- without MATE the downsample output is byte-identical to the historical uniform row sampler.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import polars as pl
from ugbio_featuremap import featuremap_to_dataframe
from ugbio_featuremap.featuremap_to_dataframe import (
    COL_MATE,
    SAMPLE_WEIGHT_COL,
    _is_multi_call_schema,
    _merge_parquet_files_lazy,
    _multi_call_molecule_key_cols,
)
from ugbio_featuremap.featuremap_utils import FeatureMapFields

_HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=chr1,length=1000000>\n"
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
    '##FORMAT=<ID=RN,Number=.,Type=String,Description="Read name">\n'
    '##FORMAT=<ID=REV,Number=.,Type=Integer,Description="Is reverse strand">\n'
)
_MATE_HEADER = '##FORMAT=<ID=MATE,Number=.,Type=Integer,Description="Multi-call mate: 0 unpaired, 1 R1, 2 R2">\n'


def _write_vcf(tmp_path: Path, name: str, *, with_mate: bool) -> Path:
    fmt = "GT:RN:REV:MATE" if with_mate else "GT:RN:REV"
    s1 = "0/1:q1,q1:0,0:1,2" if with_mate else "0/1:q1,q2:0,1"
    s2 = "0/1:q3:1:2" if with_mate else "0/1:q3:1"
    txt = (
        _HEADER
        + (_MATE_HEADER if with_mate else "")
        + "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\n"
        + f"chr1\t100\t.\tA\tG\t30.0\tPASS\t.\t{fmt}\t{s1}\n"
        + f"chr1\t200\t.\tC\tT\t30.0\tPASS\t.\t{fmt}\t{s2}\n"
    )
    plain = tmp_path / f"{name}.vcf"
    plain.write_text(txt)
    vcf_gz = tmp_path / f"{name}.vcf.gz"
    subprocess.run(["bcftools", "view", str(plain), "-Oz", "-o", str(vcf_gz)], check=True)
    subprocess.run(["bcftools", "index", "-t", str(vcf_gz)], check=True)
    return vcf_gz


def test_mate_enum_field():
    assert FeatureMapFields.MATE.value == "MATE" == COL_MATE


def test_mate_carried_into_parquet(tmp_path: Path):
    vcf_gz = _write_vcf(tmp_path, "mate", with_mate=True)
    out = tmp_path / "mate.parquet"
    featuremap_to_dataframe.vcf_to_parquet(str(vcf_gz), str(out), drop_format={"GT"}, jobs=1)
    frame = pl.read_parquet(out).sort("POS", "MATE")
    assert COL_MATE in frame.columns
    assert frame["MATE"].dtype.is_integer()
    assert frame.select("POS", "RN", "REV", "MATE").rows() == [
        (100, "q1", 0, 1),
        (100, "q1", 0, 2),
        (200, "q3", 1, 2),
    ]


def test_no_mate_vcf_has_no_mate_column(tmp_path: Path):
    vcf_gz = _write_vcf(tmp_path, "nomate", with_mate=False)
    out = tmp_path / "nomate.parquet"
    featuremap_to_dataframe.vcf_to_parquet(str(vcf_gz), str(out), drop_format={"GT"}, jobs=1)
    frame = pl.read_parquet(out)
    assert COL_MATE not in frame.columns
    assert frame.height == 3  # noqa: PLR2004
    assert not _is_multi_call_schema(frame.columns)


# ──────────────────────────── downsampling ────────────────────────────


def _multi_call_parts(tmp_path: Path, n_parts: int = 3, n_mol: int = 400, *, with_mate: bool = True) -> list[str]:
    """Part-files of multi-call rows: ~half the molecules concordant (MATE 1 + 2 rows), the rest single-call.
    Every 5th molecule is an indel. Molecules never span parts (each part = its own POS range)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    files = []
    for p in range(n_parts):
        rows = []
        for m in range(n_mol):
            pos = p * 100_000 + m * 10
            is_indel = m % 5 == 0
            base = {
                "CHROM": "chr1",
                "POS": pos,
                "REF": "A",
                "ALT": "AC" if is_indel else "G",
                "X_IC": "ins" if is_indel else None,
                "RN": f"read_{p}_{m}",
                "REV": m % 2,
                "SNVQ": float(m),
            }
            mates = [1, 2] if m % 2 == 0 else [1 + (m % 3 == 0)]
            for mate in mates:
                row = dict(base)
                if with_mate:
                    row["MATE"] = mate
                rows.append(row)
        f = tmp_path / f"part{p}{'' if with_mate else '_nomate'}.parquet"
        pl.DataFrame(rows).write_parquet(f)
        files.append(str(f))
    return files


def _assert_molecules_whole(full: pl.DataFrame, out: pl.DataFrame) -> None:
    key = _multi_call_molecule_key_cols(full.columns)
    full_sizes = full.group_by(key).len().rename({"len": "n_full"})
    out_sizes = out.group_by(key).len().rename({"len": "n_out"})
    joined = out_sizes.join(full_sizes, on=key, how="left", nulls_equal=True)
    assert joined["n_full"].null_count() == 0
    assert (joined["n_out"] == joined["n_full"]).all(), "a molecule's two calls were split by the downsample"


def test_multi_call_downsample_keeps_mates_together(tmp_path: Path):
    files = _multi_call_parts(tmp_path)
    full = pl.concat([pl.read_parquet(f) for f in files])
    out = tmp_path / "out.parquet"
    _merge_parquet_files_lazy(files, str(out), downsample_reads=500, downsample_seed=7)
    res = pl.read_parquet(out)
    assert 490 <= res.height <= 500  # noqa: PLR2004 - whole-molecule trim is short by at most one molecule
    _assert_molecules_whole(full, res)
    # both concordant and single-call molecules survive
    sizes = res.group_by("RN").len()["len"]
    assert (sizes == 2).any() and (sizes == 1).any()  # noqa: PLR2004
    assert SAMPLE_WEIGHT_COL in res.columns
    assert "_mol_key" not in res.columns


def test_multi_call_downsample_seed_reproducible(tmp_path: Path):
    outs = []
    for k in range(2):
        files = _multi_call_parts(tmp_path / f"r{k}")
        out = tmp_path / f"out{k}.parquet"
        _merge_parquet_files_lazy(files, str(out), downsample_reads=300, downsample_seed=3)
        outs.append(pl.read_parquet(out))
    assert outs[0].equals(outs[1])


def test_multi_call_downsample_with_indel_balance(tmp_path: Path):
    files = _multi_call_parts(tmp_path / "a")
    full = pl.concat([pl.read_parquet(f) for f in files])
    out = tmp_path / "out.parquet"
    _merge_parquet_files_lazy(files, str(out), downsample_reads=400, downsample_seed=1, indel_snv_balance_fraction=0.5)
    res = pl.read_parquet(out)
    _assert_molecules_whole(full, res)
    n_indel = res.filter(pl.col("X_IC").is_in(["ins", "del"]).fill_null(value=False)).height
    assert 180 <= n_indel <= 200  # noqa: PLR2004
    assert res.height <= 400  # noqa: PLR2004


def test_multi_call_downsample_ignores_duplex_priority(tmp_path: Path):
    files = _multi_call_parts(tmp_path)
    full = pl.concat([pl.read_parquet(f) for f in files])
    for f in files:  # a CS column would normally route to the duplex priority path
        pl.read_parquet(f).with_columns(pl.col("RN").alias("CS"), nf=pl.lit(0), nr=pl.lit(0)).write_parquet(f)
    out = tmp_path / "out.parquet"
    _merge_parquet_files_lazy(
        files, str(out), downsample_reads=300, downsample_seed=2, duplex_oversample_priority=["duplex_pe_full"]
    )
    res = pl.read_parquet(out)
    _assert_molecules_whole(full, res.drop("CS", "nf", "nr"))
    assert "duplex_group" not in res.columns


def test_no_mate_downsample_unchanged(tmp_path: Path):
    """Without MATE the uniform multi-part sampler runs exactly as before (reference re-implemented here)."""
    files_ref = _multi_call_parts(tmp_path / "ref", with_mate=False)
    files_new = _multi_call_parts(tmp_path / "new", with_mate=False)
    n_rows, seed = 300, 11
    total = sum(pl.read_parquet(f).height for f in files_ref)
    keep_frac = min(1.0, (n_rows / total) * 1.05 + 1000.0 / total)
    expected = pl.concat(
        [pl.read_parquet(f).sample(fraction=keep_frac, seed=seed + i) for i, f in enumerate(files_ref)]
    ).sample(n=n_rows, seed=seed)

    out = tmp_path / "out.parquet"
    _merge_parquet_files_lazy(files_new, str(out), downsample_reads=n_rows, downsample_seed=seed)
    res = pl.read_parquet(out)
    assert res.drop(SAMPLE_WEIGHT_COL).equals(expected)
