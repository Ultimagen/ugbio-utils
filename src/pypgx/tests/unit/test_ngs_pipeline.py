import subprocess
import threading
import time
import zipfile
from pathlib import Path

import pytest
from ugbio_pypgx import ngs_pipeline
from ugbio_pypgx.ngs_pipeline import PROFILE_KINDS, TSV_HEADER, ZIP_KINDS, run


def write_gene_outputs(command: list) -> int:
    """Gene name picks the outcome: FAIL* exits 1, BARE* writes only results.zip, NODATA* has no
    data.tsv, EMPTY* an empty one, CORRUPT* a results.zip that is not a zip."""
    gene, out_dir = command[2], Path(command[3])
    if gene.startswith("FAIL"):
        return 1
    out_dir.mkdir(parents=True, exist_ok="--force" in command)
    if not gene.startswith("BARE"):
        for kind in ZIP_KINDS:
            (out_dir / f"{kind}.zip").write_text(f"{kind} of {gene}")
        for profile in PROFILE_KINDS:
            (out_dir / profile).mkdir()
            for name in ("b.png", "a.png"):
                (out_dir / profile / name).write_text(f"{profile}/{name} of {gene}")
    if gene.startswith("CORRUPT"):
        (out_dir / "results.zip").write_text("truncated")
        return 0
    with zipfile.ZipFile(out_dir / "results.zip", "w") as archive:
        if gene.startswith("EMPTY"):
            archive.writestr("uuid/data.tsv", "\n")
        elif not gene.startswith("NODATA"):
            archive.writestr("uuid/data.tsv", f"\tGenotype\tPhenotype\nS1\tfirst-{gene}\tx\nS1\tlast-{gene}\ty\n\n")
    return 0


class FakePypgx:
    def __init__(self):
        self.commands: list[list] = []
        self.delay = 0.0
        self.max_running = 0
        self._running = 0
        self._lock = threading.Lock()

    def __call__(self, command, stdout, stderr, check):
        with self._lock:
            self.commands.append(command)
            self._running += 1
            self.max_running = max(self.max_running, self._running)
        time.sleep(self.delay)
        stdout.write(f"output of {command[2]}\n".encode())
        if stderr is subprocess.STDOUT:
            stdout.write(f"errors of {command[2]}\n".encode())
        with self._lock:
            self._running -= 1
        returncode = write_gene_outputs(command)
        if check and returncode:
            raise subprocess.CalledProcessError(returncode, command)
        return subprocess.CompletedProcess(command, returncode)


@pytest.fixture
def fake_pypgx(monkeypatch):
    fake = FakePypgx()
    monkeypatch.setattr(ngs_pipeline.subprocess, "run", fake)
    monkeypatch.setattr(ngs_pipeline.shutil, "which", lambda name: "/usr/bin/pypgx" if name == "pypgx" else None)
    return fake


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


def test_runs_parallel_genes_at_a_time(fake_pypgx, tmp_path):
    fake_pypgx.delay = 0.2

    assert run_pipeline(tmp_path, ["G1", "G2", "G3", "G4", "G5"], parallel=2) == 0

    assert fake_pypgx.max_running == 2


def test_writes_wdl_named_outputs_and_tsv_of_last_rows(fake_pypgx, tmp_path):
    assert run_pipeline(tmp_path, ["CYP2D6", "BARE1"]) == 0

    out = tmp_path / "out"
    for kind in ZIP_KINDS:
        assert (out / f"{kind}.CYP2D6.zip").read_text() == f"{kind} of CYP2D6"
        assert (out / f"{kind}.BARE1.zip").read_bytes() == b""
    for profile in PROFILE_KINDS:
        assert (out / f"{profile}.CYP2D6.png").read_text() == f"{profile}/a.png of CYP2D6"
        assert (out / f"{profile}.BARE1.png").read_bytes() == b""
    assert (out / "sample.pypgx.tsv").read_text() == (
        f"{TSV_HEADER}\nCYP2D6\tS1\tlast-CYP2D6\ty\nBARE1\tS1\tlast-BARE1\ty\n"
    )


def test_runs_pypgx_with_force_and_logs_its_output(fake_pypgx, tmp_path):
    (tmp_path / "ngs" / "G1").mkdir(parents=True)

    assert run_pipeline(tmp_path, ["G1"]) == 0

    assert fake_pypgx.commands == [
        [
            "pypgx",
            "run-ngs-pipeline",
            "G1",
            tmp_path / "ngs" / "G1",
            *["--assembly", "GRCh38", "--variants", "PASS.vcf.gz"],
            *["--depth-of-coverage", "depth.zip", "--control-statistics", "control.zip"],
            "--force",
        ]
    ]
    assert (tmp_path / "logs" / "ngs.G1.log").read_text() == "output of G1\nerrors of G1\n"


def test_failed_gene_lets_others_finish_and_exits_non_zero(fake_pypgx, tmp_path, capsys):
    assert run_pipeline(tmp_path, ["FAIL1", "G2", "G3"], parallel=1) == 1

    assert sorted(command[2] for command in fake_pypgx.commands) == ["FAIL1", "G2", "G3"]
    assert (tmp_path / "ngs" / "G3" / "results.zip").is_file()
    assert not list((tmp_path / "out").iterdir())
    assert capsys.readouterr().err == f"pypgx run-ngs-pipeline failed for:\n  FAIL1: {tmp_path}/logs/ngs.FAIL1.log\n"


@pytest.mark.parametrize(
    ("gene", "error"),
    [("NODATA1", "no data.tsv"), ("EMPTY1", "empty data.tsv"), ("CORRUPT1", "CORRUPT1/results.zip is not a zip")],
)
def test_unreadable_results_fail_with_the_gene_and_its_log(fake_pypgx, tmp_path, capsys, gene, error):
    assert run_pipeline(tmp_path, ["G1", gene]) == 1

    err = capsys.readouterr().err
    assert err.startswith(f"pypgx results unreadable for:\n  {gene}: ")
    assert err.endswith(f"; log {tmp_path}/logs/ngs.{gene}.log\n")
    assert error in err
    assert "G1:" not in err
    assert not list((tmp_path / "out").iterdir())


def test_duplicate_genes_are_rejected_before_any_gene_runs(fake_pypgx, tmp_path, capsys):
    with pytest.raises(SystemExit, match="2"):
        run_pipeline(tmp_path, ["G1", "G2", "G1", "G3", "G2"])

    assert "duplicate genes: G1 G2" in capsys.readouterr().err
    assert not fake_pypgx.commands


@pytest.mark.parametrize("parallel", [0, -1])
def test_parallel_must_be_positive(fake_pypgx, tmp_path, capsys, parallel):
    with pytest.raises(SystemExit, match="2"):
        run_pipeline(tmp_path, ["G1"], parallel=parallel)

    assert f"--parallel: must be at least 1, got {parallel}" in capsys.readouterr().err
    assert not fake_pypgx.commands


def test_missing_pypgx_fails_before_any_gene_runs(fake_pypgx, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(ngs_pipeline.shutil, "which", lambda name: None)

    assert run_pipeline(tmp_path, ["G1"]) == 1

    assert capsys.readouterr().err == "pypgx is not on PATH\n"
    assert not fake_pypgx.commands


def test_tsv_directory_is_created(fake_pypgx, tmp_path):
    flags = pipeline_flags(tmp_path, ["G1"])
    flags["--tsv"] = [str(tmp_path / "tsv" / "sample" / "sample.pypgx.tsv")]

    assert run_flags(flags) == 0

    assert (tmp_path / "tsv" / "sample" / "sample.pypgx.tsv").read_text().startswith(TSV_HEADER)


@pytest.mark.parametrize("flag", list(pipeline_flags(Path("tmp"), ["G1"])))
def test_every_flag_is_required(fake_pypgx, tmp_path, flag):
    flags = pipeline_flags(tmp_path, ["G1"])
    del flags[flag]

    with pytest.raises(SystemExit, match="2"):
        run_flags(flags)

    assert not fake_pypgx.commands
