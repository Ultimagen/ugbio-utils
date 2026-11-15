import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parent.parent / "trivy_json_to_summary.py"
spec = importlib.util.spec_from_file_location("trivy_json_to_summary", SCRIPT)
summary_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary_mod)

SECRET_MATCH = "github_token=ghp_****************************"

SECRET_RESULT = {
    "Target": "/app/.env",
    "Class": "secret",
    "Secrets": [
        {
            "RuleID": "github-pat",
            "Category": "GitHub",
            "Severity": "CRITICAL",
            "Title": "GitHub Personal Access Token",
            "StartLine": 3,
            "Match": SECRET_MATCH,
            "Layer": {"DiffID": "sha256:abc"},
        }
    ],
}

HIGH_VULN_RESULT = {
    "Target": "debian",
    "Class": "os-pkgs",
    "Vulnerabilities": [
        {"VulnerabilityID": "CVE-1", "PkgName": "perl", "InstalledVersion": "1", "FixedVersion": "", "Severity": "HIGH"}
    ],
}

BLOCKING_VULN_RESULT = {
    "Target": "debian",
    "Class": "os-pkgs",
    "Vulnerabilities": [
        {
            "VulnerabilityID": "CVE-2",
            "PkgName": "zlib",
            "InstalledVersion": "1",
            "FixedVersion": "2",
            "Severity": "CRITICAL",
        }
    ],
}


def _write(tmp_path: Path, results: list[dict]) -> Path:
    path = tmp_path / "trivy-results.json"
    path.write_text(json.dumps({"Results": results}))
    return path


def _run(tmp_path: Path, results: list[dict], policy: str) -> tuple[subprocess.CompletedProcess, dict, str]:
    results_file = _write(tmp_path, results)
    outputs = tmp_path / "outputs"
    summary = tmp_path / "summary.md"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--results-file",
            str(results_file),
            "--image-name",
            "img",
            "--image-tag",
            "tag",
            "--policy",
            policy,
            "--github-step-summary",
            str(summary),
        ],
        env={"GITHUB_OUTPUT": str(outputs)},
        capture_output=True,
        text=True,
        check=False,
    )
    parsed = dict(line.split("=", 1) for line in outputs.read_text().splitlines())
    return proc, parsed, summary.read_text() if summary.exists() else ""


def test_parse_secrets_drops_the_match(tmp_path):
    data = summary_mod.parse_trivy_json(_write(tmp_path, [SECRET_RESULT]))
    assert data["secrets"] == [
        {
            "rule": "github-pat",
            "title": "GitHub Personal Access Token",
            "severity": "CRITICAL",
            "location": "/app/.env",
            "line": 3,
            "layer": "sha256:abc",
        }
    ]
    assert data["counts"] == {}


@pytest.mark.parametrize("policy", ["warn", "block"])
def test_secret_blocks_under_any_policy(tmp_path, policy):
    proc, outputs, summary = _run(tmp_path, [SECRET_RESULT, HIGH_VULN_RESULT], policy)
    assert proc.returncode == 1
    assert outputs["blocked"] == "true"
    assert outputs["secret-count"] == "1"
    assert "github-pat" in summary
    assert "push blocked" in summary
    assert "::error title=Secret github-pat in image::/app/.env" in proc.stdout
    assert SECRET_MATCH not in proc.stdout + summary


def test_secret_reported_when_there_are_no_vulnerabilities(tmp_path):
    proc, _, summary = _run(tmp_path, [SECRET_RESULT], "warn")
    assert proc.returncode == 1
    assert "github-pat" in summary
    assert "No vulnerabilities found" in summary


def test_no_secrets_keeps_warn_policy_non_blocking(tmp_path):
    proc, outputs, summary = _run(tmp_path, [BLOCKING_VULN_RESULT], "warn")
    assert proc.returncode == 0
    assert outputs["blocked"] == "false"
    assert outputs["secret-count"] == "0"
    assert "secrets detected" not in summary


def test_no_secrets_block_policy_still_blocks_on_cve(tmp_path):
    proc, outputs, _ = _run(tmp_path, [BLOCKING_VULN_RESULT], "block")
    assert proc.returncode == 1
    assert outputs["blocked"] == "true"
    assert outputs["blocking-count"] == "1"
