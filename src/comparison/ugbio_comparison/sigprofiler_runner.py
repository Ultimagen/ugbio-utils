"""Run the SigProfiler signature analysis, which lives in its own virtual environment.

SigProfiler needs numpy>=2 while the ugbio workspace pins numpy<2, so the analysis
(``src/comparison/sigprofiler``, package ``ugbio_sigprofiler``) is a separate project
with its own virtual environment. This module finds its Python interpreter and runs it:

1. ``$UGBIO_SIGPROFILER_PYTHON`` if set (the ugbio_comparison Docker image sets it)
2. otherwise, in a repository checkout, ``uv run --project src/comparison/sigprofiler``,
   which creates or updates that project's ``.venv`` on first use
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
from ugbio_core.logger import logger

SIGPROFILER_PYTHON_ENV = "UGBIO_SIGPROFILER_PYTHON"
SIGPROFILER_PROJECT_DIR = Path(__file__).resolve().parents[1] / "sigprofiler"
SIGPROFILER_MODULE = "ugbio_sigprofiler.somatic_signatures"
SIGNATURE_CONTEXTS = ("SBS96", "ID83", "DBS78")
FINAL_RESULTS_DIR = "final_results"
SAMPLE_NAME_KEY = "signatures_sample_name"


class SigProfilerUnavailableError(RuntimeError):
    """Neither the SigProfiler interpreter nor the SigProfiler project can be found"""


def sigprofiler_command() -> list[str]:
    """Command prefix that runs the ``somatic_signatures`` module in the SigProfiler environment

    Raises
    ------
    SigProfilerUnavailableError
        If there is no SigProfiler interpreter and no ``uv`` with the SigProfiler project
    """
    python = os.environ.get(SIGPROFILER_PYTHON_ENV)
    if python:
        return [python, "-m", SIGPROFILER_MODULE]
    uv = shutil.which("uv")
    if uv and (SIGPROFILER_PROJECT_DIR / "pyproject.toml").is_file():
        return [uv, "run", "--quiet", "--project", str(SIGPROFILER_PROJECT_DIR), "python", "-m", SIGPROFILER_MODULE]
    raise SigProfilerUnavailableError(
        f"SigProfiler environment not found: set {SIGPROFILER_PYTHON_ENV} to its Python interpreter, "
        f"or run from a repository checkout with uv installed ({SIGPROFILER_PROJECT_DIR})"
    )


def run_somatic_signatures(output_dir: Path, sigprofiler_args: list[str]) -> dict:
    """Run ``somatic_signatures`` in the SigProfiler environment.

    Parameters
    ----------
    output_dir : Path
        Output directory of the analysis
    sigprofiler_args : list[str]
        Other ``somatic_signatures`` arguments (input, genome, COSMIC version)

    Returns
    -------
    dict
        The analysis summary (``summary.json``): sample name and the contexts that had mutations
    """
    cmd = [*sigprofiler_command(), "--output_dir", str(output_dir), *sigprofiler_args]
    # The workspace virtual environment must not leak into the SigProfiler one
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    logger.info(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)
    with open(output_dir / "summary.json") as summary_file:
        return json.load(summary_file)


def save_signatures_to_h5(output_dir: Path, summary: dict, h5_file: str) -> None:
    """Store the analysis results in an h5 file.

    Keys: ``<context>_activity`` (activity per assigned signature), ``<context>_profile`` (sample mutation
    matrix) for every context with mutations, and ``signatures_sample_name`` (the name in the plot file names).
    """
    # to_hdf appends: drop the signatures of contexts that are absent from this run
    if Path(h5_file).exists():
        new_keys = {f"/{context}_{kind}" for context in summary["contexts"] for kind in ("activity", "profile")}
        with pd.HDFStore(h5_file) as store:
            for key in store.keys():
                is_signature_key = any(key in (f"/{c}_activity", f"/{c}_profile") for c in SIGNATURE_CONTEXTS)
                if is_signature_key and key not in new_keys:
                    store.remove(key)
    pd.Series([summary["sample_name"]]).to_hdf(h5_file, key=SAMPLE_NAME_KEY)
    for context in summary["contexts"]:
        activities = pd.read_csv(output_dir / f"{context}_activity.tsv", sep="\t", index_col=0)["activity"]
        activities.index.name = None
        activities.to_hdf(h5_file, key=f"{context}_activity")
        profile = pd.read_csv(output_dir / f"{context}_profile.tsv", sep="\t")
        profile.to_hdf(h5_file, key=f"{context}_profile")
