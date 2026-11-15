import importlib.util
import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "binary_findings.py"
spec = importlib.util.spec_from_file_location("binary_findings", SCRIPT)
bf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bf)

PYTHON_BINARY = {
    "type": "binary",
    "name": "python",
    "version": "3.12.13",
    "locations": [{"path": "/usr/local/bin/python3.12"}, {"path": "/usr/local/lib/libpython3.12.so.1.0"}],
}


def _match(artifact: dict, cve: str, severity: str, fixed: list[str]) -> dict:
    return {
        "artifact": artifact,
        "vulnerability": {"id": cve, "severity": severity, "fix": {"versions": fixed, "state": "fixed"}},
    }


GRYPE = {
    "matches": [
        _match(PYTHON_BINARY, "CVE-2026-6100", "High", ["3.12.14"]),
        _match(PYTHON_BINARY, "CVE-2026-19445", "Critical", ["3.12.15"]),
        _match(PYTHON_BINARY, "CVE-2026-0001", "Medium", []),
        _match({"type": "deb", "name": "zlib1g", "version": "1", "locations": []}, "CVE-2026-9999", "Critical", ["2"]),
        _match({"type": "python", "name": "requests", "version": "2", "locations": []}, "CVE-2026-8888", "High", []),
    ]
}


def _file(path: str, *, unknown: bool) -> dict:
    entry = {"location": {"path": path}}
    if unknown:
        entry["unknowns"] = ["unknowns-labeler: no package identified in executable file"]
    return entry


SBOM = {
    "files": [
        _file("/usr/local/bin/samtools", unknown=True),
        _file("/opt/tool/bin/bwa", unknown=True),
        _file("/usr/local/lib/python3.12/lib-dynload/_ssl.cpython-312-x86_64-linux-gnu.so", unknown=True),
        _file("/usr/local/lib/python3.12/site-packages/numpy/core/_multiarray.so", unknown=True),
        _file("/usr/lib/python3/dist-packages/apt_pkg.so", unknown=True),
        _file("/app/node_modules/sharp/build/sharp.node", unknown=True),
        _file("/usr/local/lib/libpython3.so", unknown=True),
        _file("/usr/bin/bash", unknown=False),
    ]
}


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_parse_grype_keeps_only_binary_packages_sorted_by_severity(tmp_path):
    findings = bf.parse_grype(_write(tmp_path, "grype.json", GRYPE))
    assert [(f["cve"], f["severity"]) for f in findings] == [
        ("CVE-2026-19445", "Critical"),
        ("CVE-2026-6100", "High"),
        ("CVE-2026-0001", "Medium"),
    ]
    assert findings[1]["fixed"] == "3.12.14"
    assert findings[2]["fixed"] == "NotAvailable"
    assert findings[0]["location"] == "/usr/local/bin/python3.12, /usr/local/lib/libpython3.12.so.1.0"


def test_parse_unscanned_skips_language_packages_and_cpython_modules(tmp_path):
    assert bf.parse_unscanned(_write(tmp_path, "sbom.json", SBOM)) == ["/opt/tool/bin/bwa", "/usr/local/bin/samtools"]


def test_summary_lists_high_and_critical_only_in_details(tmp_path):
    findings = bf.parse_grype(_write(tmp_path, "grype.json", GRYPE))
    summary = bf.generate_summary("img:tag", findings, ["/usr/local/bin/samtools"])
    assert "| Critical | 1 |" in summary
    assert "| Medium | 1 |" in summary
    assert "CVE-2026-6100" in summary
    assert "CVE-2026-0001" not in summary
    assert "- `/usr/local/bin/samtools`" in summary


def test_summary_when_clean():
    summary = bf.generate_summary("img:tag", [], [])
    assert "No vulnerabilities found in binaries" in summary
    assert "Every executable belongs to a known package" in summary


def test_summary_when_results_missing():
    summary = bf.generate_summary("img:tag", None, None)
    assert "Grype results not available" in summary
    assert "SBOM not available" in summary


def _run(args: list[str], summary: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args, "--image", "img:tag", "--github-step-summary", str(summary)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_reports_without_failing(tmp_path):
    summary = tmp_path / "summary.md"
    grype = _write(tmp_path, "grype.json", GRYPE)
    sbom = _write(tmp_path, "sbom.json", SBOM)
    proc = _run(["--grype-results", str(grype), "--sbom", str(sbom)], summary)
    assert proc.returncode == 0
    assert "::warning title=Binary CVE-2026-19445::python 3.12.13" in proc.stdout
    assert "::warning title=Unscanned executables in img:tag::2 executables" in proc.stdout
    assert "Binaries not covered by Trivy" in summary.read_text()


def test_cli_missing_inputs_warns_and_passes(tmp_path):
    summary = tmp_path / "summary.md"
    proc = _run(["--grype-results", str(tmp_path / "none.json"), "--sbom", str(tmp_path / "none2.json")], summary)
    assert proc.returncode == 0
    assert "::warning::Grype results not found" in proc.stdout
    assert "::warning::SBOM not found" in proc.stdout
