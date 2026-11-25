import json
from pathlib import Path

import pandas as pd
import pytest
from ugbio_comparison import sigprofiler_runner

MATRICES_DIR = Path(__file__).parent.parent / "resources" / "sigprofiler_matrices"


def test_sigprofiler_command_from_env(monkeypatch):
    monkeypatch.setenv(sigprofiler_runner.SIGPROFILER_PYTHON_ENV, "/opt/sigprofiler/bin/python")
    assert sigprofiler_runner.sigprofiler_command() == [
        "/opt/sigprofiler/bin/python",
        "-m",
        sigprofiler_runner.SIGPROFILER_MODULE,
    ]


def test_sigprofiler_command_from_project(monkeypatch, tmp_path):
    monkeypatch.delenv(sigprofiler_runner.SIGPROFILER_PYTHON_ENV, raising=False)
    monkeypatch.setattr(sigprofiler_runner.shutil, "which", lambda name: f"/usr/bin/{name}")
    (tmp_path / "pyproject.toml").touch()
    monkeypatch.setattr(sigprofiler_runner, "SIGPROFILER_PROJECT_DIR", tmp_path)
    assert sigprofiler_runner.sigprofiler_command() == [
        "/usr/bin/uv",
        "run",
        "--quiet",
        "--project",
        str(tmp_path),
        "python",
        "-m",
        sigprofiler_runner.SIGPROFILER_MODULE,
    ]


def test_sigprofiler_project_dir_in_checkout():
    comparison_dir = Path(__file__).parents[2]
    if not Path(sigprofiler_runner.__file__).resolve().is_relative_to(comparison_dir.resolve()):
        pytest.skip("ugbio_comparison is not installed from this checkout")
    assert sigprofiler_runner.SIGPROFILER_PROJECT_DIR == (comparison_dir / "sigprofiler").resolve()
    assert (sigprofiler_runner.SIGPROFILER_PROJECT_DIR / "pyproject.toml").is_file()


def test_sigprofiler_command_unavailable(monkeypatch, tmp_path):
    monkeypatch.delenv(sigprofiler_runner.SIGPROFILER_PYTHON_ENV, raising=False)
    monkeypatch.setattr(sigprofiler_runner, "SIGPROFILER_PROJECT_DIR", tmp_path)
    with pytest.raises(sigprofiler_runner.SigProfilerUnavailableError):
        sigprofiler_runner.sigprofiler_command()


def test_save_signatures_to_h5(tmp_path):
    pd.DataFrame({"signature": ["SBS1", "SBS5"], "activity": [120.0, 30.0]}).to_csv(
        tmp_path / "SBS96_activity.tsv", sep="\t", index=False
    )
    profile = pd.DataFrame({"MutationType": ["A[C>A]A", "A[C>A]C"], "sample": [3, 0]})
    profile.to_csv(tmp_path / "SBS96_profile.tsv", sep="\t", index=False)
    h5_file = str(tmp_path / "stats.h5")

    sigprofiler_runner.save_signatures_to_h5(tmp_path, {"sample_name": "sample", "contexts": ["SBS96"]}, h5_file)

    activities = pd.read_hdf(h5_file, "SBS96_activity")
    assert activities.to_dict() == {"SBS1": 120.0, "SBS5": 30.0}
    assert activities.index.name is None
    pd.testing.assert_frame_equal(pd.read_hdf(h5_file, "SBS96_profile"), profile)
    assert pd.read_hdf(h5_file, sigprofiler_runner.SAMPLE_NAME_KEY).iloc[0] == "sample"


@pytest.fixture
def sigprofiler_available():
    try:
        sigprofiler_runner.sigprofiler_command()
    except sigprofiler_runner.SigProfilerUnavailableError as e:
        pytest.skip(str(e))


@pytest.mark.usefixtures("sigprofiler_available")
def test_run_somatic_signatures_from_matrices(tmp_path):
    output_dir = tmp_path / "somatic"
    summary = sigprofiler_runner.run_somatic_signatures(output_dir, ["--matrices_dir", str(MATRICES_DIR)])

    assert summary == {"sample_name": "somatic_test", "contexts": ["SBS96", "ID83", "DBS78"]}
    assert json.loads((output_dir / "summary.json").read_text()) == summary
    final_results = output_dir / sigprofiler_runner.FINAL_RESULTS_DIR
    for context, png_prefix in (("SBS96", "SBS_96"), ("ID83", "ID_83"), ("DBS78", "DBS_78")):
        assert (final_results / f"{png_prefix}_plots_somatic_test.png").is_file()
        activities = pd.read_csv(output_dir / f"{context}_activity.tsv", sep="\t", index_col=0)["activity"]
        assert not activities.empty
        assert (activities > 0).all()
        for signature in activities.index:
            assert (final_results / f"{png_prefix}_plots_{signature}.png").is_file()

    # The somatic test sample is dominated by the clock-like signatures
    sbs_activities = pd.read_csv(output_dir / "SBS96_activity.tsv", sep="\t", index_col=0)["activity"]
    assert {"SBS1", "SBS5"} <= set(sbs_activities.index)

    h5_file = str(tmp_path / "stats.h5")
    sigprofiler_runner.save_signatures_to_h5(output_dir, summary, h5_file)
    for context in summary["contexts"]:
        assert not pd.read_hdf(h5_file, f"{context}_activity").empty
        assert not pd.read_hdf(h5_file, f"{context}_profile").empty
