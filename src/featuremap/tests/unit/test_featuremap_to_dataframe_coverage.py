"""featuremap_to_dataframe: in-process region conversion, CLI, and downsample edge cases.

The region workers normally run in spawned processes (invisible to coverage); here they are also run in-process
and must produce exactly what the spawned path produces.
"""

import json
import logging
import subprocess
from pathlib import Path

import polars as pl
import pytest
from ugbio_featuremap import featuremap_to_dataframe as f2d
from ugbio_featuremap.featuremap_to_dataframe import (
    DUPLEX_GROUP_COL,
    DUPLEX_GROUP_DUPLEX,
    DUPLEX_GROUP_DUPLEX_PE_FULL,
    DUPLEX_GROUP_SINGLETON,
    DUPLEX_GROUP_SSC,
    DUPLEX_PRIORITY_FP,
)

RESOURCES = Path(__file__).parent.parent / "resources"
FEATUREMAP = RESOURCES / "23A03846_bc_30.head.featuremap.vcf.gz"
READ_FILTERS = RESOURCES / "create_featuremap_read_filters_test.simple.json"


def _serial_region_jobs(*, regions, vcf_path, fmt_str, job_cfg, jobs, tmpdir, spawn_ctx):  # noqa: ARG001
    """In-process stand-in for _run_region_jobs (same part-file naming, same skip-empty semantics)."""
    out = []
    for i, region in enumerate(regions):
        part = str(Path(tmpdir) / f"part_{i:06d}.parquet")
        if f2d._process_region_to_parquet(region, vcf_path, fmt_str, job_cfg, part):
            out.append(part)
    return out


@pytest.fixture
def in_process(monkeypatch):
    monkeypatch.setattr(f2d, "_run_region_jobs", _serial_region_jobs)
    # worker-side logging setup would reconfigure the test process' root logger
    monkeypatch.setattr(f2d, "_configure_logging", lambda *a, **k: None)


def _convert(tmp_path, name, vcf, **kwargs):
    out = str(tmp_path / f"{name}.parquet")
    f2d.vcf_to_parquet(str(vcf), out, jobs=1, **kwargs)
    return pl.read_parquet(out)


def _sorted(frame):
    return frame.sort([c for c in ("CHROM", "POS", "RN") if c in frame.columns])


def _multi_sample_vcf(tmp_path):
    vcf_txt = (
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr19,length=58617616>\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
        '##FORMAT=<ID=MQ,Number=.,Type=Integer,Description="Per-read mapq">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
        "chr19\t1000\t.\tG\tC\t40\tPASS\t.\tGT:DP:MQ\t./.:5:60,0,30\t./.:4:10\n"
        "chr19\t2000\t.\tA\tG\t41\tPASS\t.\tGT:DP:MQ\t./.:7:60\t./.:3:.\n"
    )
    plain = tmp_path / "ms.vcf"
    plain.write_text(vcf_txt)
    gz = tmp_path / "ms.vcf.gz"
    subprocess.run(["bcftools", "view", str(plain), "-Oz", "-o", str(gz), "--write-index=tbi"], check=True)
    return gz


# ---------------------------------------------------------------- in-process == spawned


def test_in_process_conversion_matches_spawned_with_read_filters(tmp_path, monkeypatch):
    kw = {
        "drop_format": {"GT", "AD"},
        "read_filters_json": str(READ_FILTERS),
        "read_filter_json_key": "filters_random_sample",
    }
    spawned = _convert(tmp_path, "spawned", FEATUREMAP, **kw)
    unfiltered = _convert(tmp_path, "unfiltered", FEATUREMAP, drop_format={"GT", "AD"})
    monkeypatch.setattr(f2d, "_run_region_jobs", _serial_region_jobs)
    monkeypatch.setattr(f2d, "_configure_logging", lambda *a, **k: None)
    local = _convert(tmp_path, "local", FEATUREMAP, **kw)
    assert 0 < local.height < unfiltered.height == 3440
    assert (local["BCSQ"] >= 15).all()
    assert (local["MAPQ"] >= 10).all()
    assert _sorted(local).equals(_sorted(spawned))


def test_in_process_aggregate_multi_sample_matches_spawned(tmp_path, monkeypatch):
    vcf = _multi_sample_vcf(tmp_path)
    spawned = _convert(tmp_path, "spawned", vcf, list_mode="aggregate")
    monkeypatch.setattr(f2d, "_run_region_jobs", _serial_region_jobs)
    monkeypatch.setattr(f2d, "_configure_logging", lambda *a, **k: None)
    local = _convert(tmp_path, "local", vcf, list_mode="aggregate")
    assert local.height == 2  # one row per locus in aggregate mode
    assert {"DP_S1", "DP_S2"} <= set(local.columns)
    assert _sorted(local).equals(_sorted(spawned))


def test_in_process_all_rows_filtered_raises(tmp_path, in_process):
    filters = tmp_path / "f.json"
    rule = {"field": "BCSQ", "op": "ge", "value": 10_000, "type": "quality", "name": "impossible"}
    filters.write_text(json.dumps({"filters": [rule]}))
    with pytest.raises(RuntimeError, match="No Parquet part-files were produced"):
        _convert(tmp_path, "none", FEATUREMAP, drop_format={"GT", "AD"}, read_filters_json=str(filters))


def test_in_process_duplex_priority_rejected_before_conversion(tmp_path, in_process):
    with pytest.raises(ValueError, match="valid tiers"):
        _convert(tmp_path, "x", FEATUREMAP, downsample_reads=10, duplex_oversample_priority=["cs_alt_count"])


# ---------------------------------------------------------------- header helpers


def test_count_vcf_records_falls_back_to_none(tmp_path):
    assert f2d._count_vcf_records(str(FEATUREMAP), str(tmp_path / "no_such_bcftools")) is None


def test_count_vcf_records_reads_index():
    assert f2d._count_vcf_records(str(FEATUREMAP), "bcftools") > 0


def test_parse_vcf_contigs_requires_contig_lines(tmp_path):
    plain = tmp_path / "nocontig.vcf"
    plain.write_text(
        "##fileformat=VCFv4.2\n"
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"
    )
    with pytest.raises(RuntimeError, match="No suitable ##contig lines"):
        f2d._parse_vcf_contigs(str(plain), "bcftools")


# ---------------------------------------------------------------- downsample edge cases


def test_duplex_group_without_cs_classifies_each_row_alone():
    frame = pl.DataFrame(
        {
            "CHROM": ["chr1"] * 4,
            "POS": [10, 10, 20, 30],
            "REF": ["C"] * 4,
            "ALT": ["T"] * 4,
            "nf": [2, 0, 1, 0],
            "nr": [0, 2, 0, 0],
            "po": ["F1R2", "F2R1", "F1R2", None],
            "dx": [None, None, 1, None],
        }
    )
    # No CS -> no molecule key: two opposite-po rows of the same locus are NOT merged into one duplex molecule
    groups = f2d._add_duplex_group_column(frame)[DUPLEX_GROUP_COL].to_list()
    assert groups == [DUPLEX_GROUP_SSC, DUPLEX_GROUP_SSC, DUPLEX_GROUP_DUPLEX_PE_FULL, DUPLEX_GROUP_SINGLETON]
    with_cs = f2d._add_duplex_group_column(frame.with_columns(pl.lit("m").alias("CS")))
    assert with_cs[DUPLEX_GROUP_COL].to_list()[:2] == [DUPLEX_GROUP_DUPLEX, DUPLEX_GROUP_DUPLEX]


def test_priority_preserve_zero_budget_writes_empty_with_schema(tmp_path):
    rows = [{"CHROM": "chr1", "POS": i, "REF": "C", "ALT": "T", "CS": f"m{i}", "nf": 1, "nr": 0} for i in range(20)]
    part = tmp_path / "p0.parquet"
    pl.DataFrame(rows).write_parquet(part)
    out = tmp_path / "out.parquet"
    f2d._priority_preserve_downsample([str(part)], str(out), 0, 1, list(DUPLEX_PRIORITY_FP), None)
    res = pl.read_parquet(out)
    assert res.height == 0
    assert res.columns == ["CHROM", "POS", "REF", "ALT", "CS", "nf", "nr"]  # helper column dropped
    assert not part.exists()  # inputs and the annotated temporaries are cleaned up
    assert not Path(str(part) + ".dg.parquet").exists()


def test_stratified_downsample_all_indel_target_keeps_schema(tmp_path):
    frame = pl.DataFrame(
        {
            "POS": list(range(200)),
            "X_IC": ["ins"] * 100 + [None] * 100,
            "RN": [f"r{i}" for i in range(200)],
        }
    )
    files = []
    for i in range(2):
        f = tmp_path / f"p{i}.parquet"
        frame.filter(pl.col("POS") % 2 == i).write_parquet(f)
        files.append(str(f))
    out = tmp_path / "out.parquet"
    f2d._stratified_downsample_by_variant_type(files, str(out), 50, 3, 1.0)
    res = pl.read_parquet(out)
    assert res.height == 50
    assert (res["X_IC"] == "ins").all()  # SNV target is 0 -> none kept
    assert res.columns == frame.columns
    assert res["RN"].n_unique() == 50


# ---------------------------------------------------------------- CLI


@pytest.fixture
def captured_kwargs(monkeypatch):
    seen = {}
    monkeypatch.setattr(f2d, "vcf_to_parquet", lambda **kw: seen.update(kw))
    monkeypatch.setattr(f2d, "_configure_logging", lambda *a, **k: None)
    return seen


def test_cli_passes_parsed_args(captured_kwargs):
    f2d.main(
        [
            "--input",
            "in.vcf.gz",
            "--output",
            "out.parquet",
            "--jobs",
            "3",
            "--list-mode",
            "aggregate",
            "--expand-columns",
            "AD:2",
            "PL:3",
            "--downsample-reads",
            "100",
            "--downsample-seed",
            "7",
            "--indel-snv-balance-fraction",
            "0.4",
            "--duplex-oversample-priority",
            " duplex_pe_full, ssc ,",
            "--duplex-group-min-fraction",
            "0.1",
            "--per-class-counts-json",
            "c.json",
        ]
    )
    kw = captured_kwargs
    assert kw["vcf"] == "in.vcf.gz"
    assert kw["out"] == "out.parquet"
    assert kw["jobs"] == 3
    assert kw["list_mode"] == "aggregate"
    assert kw["expand_columns"] == {"AD": 2, "PL": 3}
    assert kw["downsample_reads"] == 100
    assert kw["downsample_seed"] == 7
    assert kw["indel_snv_balance_fraction"] == pytest.approx(0.4)
    assert kw["duplex_oversample_priority"] == ["duplex_pe_full", "ssc"]
    assert kw["duplex_group_min_fraction"] == pytest.approx(0.1)
    assert kw["per_class_counts_json"] == "c.json"
    assert kw["log_level"] == logging.INFO
    assert kw["drop_format"] == {"GT", "AD", "X_TCM"}


def test_cli_defaults(captured_kwargs):
    f2d.main(["--input", "in.vcf.gz", "--output", "out.parquet", "--verbose"])
    kw = captured_kwargs
    assert kw["duplex_oversample_priority"] is None
    assert kw["expand_columns"] is None
    assert kw["downsample_reads"] is None
    assert kw["log_level"] == logging.DEBUG


@pytest.mark.parametrize("spec", ["AD", "AD:two"])
def test_cli_rejects_bad_expand_columns(captured_kwargs, capsys, spec):
    with pytest.raises(SystemExit) as exc:
        f2d.main(["--input", "i", "--output", "o", "--expand-columns", spec])
    assert exc.value.code == 2
    assert "expand-columns" in capsys.readouterr().err
    assert not captured_kwargs  # never reached the conversion


# ---------------------------------------------------------------- logging


def test_configure_logging_sets_levels_and_worker_cache():
    root = logging.getLogger()
    saved = (list(root.handlers), root.level, f2d.log.level, f2d.log.propagate)
    saved_cache = getattr(f2d._configure_logging, "_worker_log_level", None)
    try:
        f2d._configure_logging(logging.WARNING)
        assert root.level == logging.WARNING
        assert f2d.log.level == logging.WARNING
        f2d._configure_logging._worker_log_level = logging.ERROR
        f2d._configure_logging(logging.ERROR, check_worker_cache=True)  # cached level -> no reconfigure
        assert root.level == logging.WARNING
        f2d._configure_logging(logging.DEBUG, check_worker_cache=True)  # level changed -> reconfigure
        assert root.level == logging.DEBUG
    finally:
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])
        f2d.log.setLevel(saved[2])
        f2d.log.propagate = saved[3]
        f2d._configure_logging._worker_log_level = saved_cache
