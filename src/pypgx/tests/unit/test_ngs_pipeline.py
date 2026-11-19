import os
from pathlib import Path

import pytest
from ugbio_pypgx.ngs_pipeline import TSV_HEADER, ZIP_KINDS, run

FAKE_PYPGX_DIR = Path(__file__).parent.parent / "resources" / "bin"


@pytest.fixture
def fake_pypgx(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", f"{FAKE_PYPGX_DIR}{os.pathsep}{os.environ['PATH']}")
    trace = tmp_path / "trace"
    trace.mkdir()
    monkeypatch.setenv("FAKE_PYPGX_TRACE", str(trace))
    return trace


def pipeline_flags(tmp_path: Path, genes: list[str], parallel: int = 2) -> dict[str, list[str]]:
    return {
        "--genes": genes,
        "--parallel": [str(parallel)],
        "--assembly": ["GRCh38"],
        "--variants": ["PASS.vcf.gz"],
        "--depth-of-coverage": ["depth.zip"],
        "--control-statistics": ["control.zip"],
        "--ngs-dir": [str(tmp_path / "ngs")],
        "--log-dir": [str(tmp_path / "logs")],
        "--output-dir": [str(tmp_path / "out")],
        "--tsv": [str(tmp_path / "out" / "sample.pypgx.tsv")],
    }


def run_flags(flags: dict[str, list[str]]) -> int:
    return run(["pgx_ngs_pipeline", *(arg for flag, values in flags.items() for arg in (flag, *values))])


def run_pipeline(tmp_path: Path, genes: list[str], parallel: int = 2) -> int:
    return run_flags(pipeline_flags(tmp_path, genes, parallel))


def max_concurrent(trace: Path) -> int:
    intervals = [tuple(map(float, path.read_text().split())) for path in trace.iterdir()]
    return max(sum(start <= t < end for start, end in intervals) for t, _ in intervals)


def test_runs_parallel_genes_at_a_time(fake_pypgx, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_PYPGX_SLEEP", "0.5")

    assert run_pipeline(tmp_path, ["G1", "G2", "G3", "G4", "G5"], parallel=2) == 0

    assert max_concurrent(fake_pypgx) == 2


def test_writes_wdl_named_outputs_and_tsv_of_last_rows(fake_pypgx, tmp_path):
    assert run_pipeline(tmp_path, ["CYP2D6", "BARE1"]) == 0

    out = tmp_path / "out"
    for kind in ZIP_KINDS:
        assert (out / f"{kind}.CYP2D6.zip").read_text() == f"{kind} of CYP2D6"
        assert (out / f"{kind}.BARE1.zip").read_bytes() == b""
    for profile in ("copy-number-profile", "allele-fraction-profile"):
        assert (out / f"{profile}.CYP2D6.png").read_text() == f"{profile}/a.png of CYP2D6"
        assert (out / f"{profile}.BARE1.png").read_bytes() == b""
    assert (out / "sample.pypgx.tsv").read_text() == (
        f"{TSV_HEADER}\nCYP2D6\tS1\tlast-CYP2D6\ty\nBARE1\tS1\tlast-BARE1\ty\n"
    )


def test_logs_each_gene_run_with_force(fake_pypgx, tmp_path):
    (tmp_path / "ngs" / "G1").mkdir(parents=True)

    assert run_pipeline(tmp_path, ["G1"]) == 0

    log = (tmp_path / "logs" / "ngs.G1.log").read_text()
    assert f"argv: run-ngs-pipeline G1 {tmp_path}/ngs/G1 --assembly GRCh38 --variants PASS.vcf.gz" in log
    assert "--depth-of-coverage depth.zip --control-statistics control.zip --force" in log
    assert "stderr of G1" in log


def test_failed_gene_lets_others_finish_and_exits_non_zero(fake_pypgx, tmp_path, capsys):
    assert run_pipeline(tmp_path, ["FAIL1", "G2", "G3"], parallel=1) == 1

    assert sorted(path.name for path in fake_pypgx.iterdir()) == ["G2", "G3"]
    assert (tmp_path / "ngs" / "G3" / "results.zip").is_file()
    assert not list((tmp_path / "out").iterdir())
    assert f"FAIL1: {tmp_path}/logs/ngs.FAIL1.log" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("gene", "error"),
    [("NODATA1", "no data.tsv"), ("EMPTY1", "empty data.tsv"), ("CORRUPT1", "CORRUPT1/results.zip is not a zip")],
)
def test_missing_or_empty_data_tsv_is_an_error(fake_pypgx, tmp_path, gene, error):
    with pytest.raises(ValueError, match=error):
        run_pipeline(tmp_path, ["G1", gene])

    assert not list((tmp_path / "out").iterdir())


def test_duplicate_genes_are_rejected_before_any_gene_runs(fake_pypgx, tmp_path, capsys):
    with pytest.raises(SystemExit, match="2"):
        run_pipeline(tmp_path, ["G1", "G2", "G1"])

    assert "duplicate genes: G1" in capsys.readouterr().err
    assert not list(fake_pypgx.iterdir())


def test_tsv_directory_is_created(fake_pypgx, tmp_path):
    flags = pipeline_flags(tmp_path, ["G1"])
    flags["--tsv"] = [str(tmp_path / "tsv" / "sample.pypgx.tsv")]

    assert run_flags(flags) == 0

    assert (tmp_path / "tsv" / "sample.pypgx.tsv").read_text().startswith(TSV_HEADER)


@pytest.mark.parametrize("flag", list(pipeline_flags(Path("tmp"), ["G1"])))
def test_every_flag_is_required(fake_pypgx, tmp_path, flag):
    flags = pipeline_flags(tmp_path, ["G1"])
    del flags[flag]

    with pytest.raises(SystemExit, match="2"):
        run_flags(flags)

    assert not list(fake_pypgx.iterdir())
