#!/usr/bin/env python3
"""
Report software that Trivy does not inventory: binaries installed outside a package manager.

Two sources, both report-only (warning annotations, exit code 0):
  - Grype matches on Syft "binary" packages, i.e. versions detected from the executable itself
    (e.g. CPython built into /usr/local by the official python images). Matches on OS and
    language packages are skipped: Trivy already reports those.
  - Executables Syft could not attribute to any package (source-built tools such as samtools).
    Nothing checks these for vulnerabilities, so they are listed for review.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

MAX_ROWS = 50
MAX_ANNOTATIONS = 10
SEVERITY_ORDER = ["Critical", "High", "Medium", "Low", "Negligible", "Unknown"]
UNKNOWN_EXECUTABLE = "no package identified in executable file"
# Language package files are scanned from their metadata; CPython's own modules belong to the interpreter
COVERED_PATHS = re.compile(r"/(site-packages|dist-packages|node_modules)/|/lib/python3\.\d+/|/libpython3[^/]*$")


def parse_grype(path: Path) -> list[dict]:
    """Grype matches on binary packages, highest severity first."""
    data = json.loads(path.read_text())
    findings = []
    for match in data.get("matches", []):
        artifact = match.get("artifact", {})
        if artifact.get("type") != "binary":
            continue
        vuln = match.get("vulnerability", {})
        fix = vuln.get("fix") or {}
        findings.append(
            {
                "cve": vuln.get("id", "N/A"),
                "package": artifact.get("name", "N/A"),
                "version": artifact.get("version", "N/A"),
                "severity": vuln.get("severity", "Unknown"),
                "fixed": ", ".join(fix.get("versions") or []) or "NotAvailable",
                "location": ", ".join(loc.get("path", "") for loc in artifact.get("locations", [])),
            }
        )
    rank = {s: i for i, s in enumerate(SEVERITY_ORDER)}
    return sorted(findings, key=lambda f: (rank.get(f["severity"], len(rank)), f["package"], f["cve"]))


def parse_unscanned(path: Path) -> list[str]:
    """Executables in the Syft SBOM that no package accounts for."""
    data = json.loads(path.read_text())
    paths = {
        f["location"]["path"]
        for f in data.get("files", [])
        if any(UNKNOWN_EXECUTABLE in u for u in f.get("unknowns") or [])
    }
    return sorted(p for p in paths if not COVERED_PATHS.search(p))


def generate_summary(image: str, findings: list[dict] | None, unscanned: list[str] | None) -> str:
    lines = [
        "",
        "---",
        "### Binaries not covered by Trivy (report only)",
        f"**Image:** `{image}`",
        "",
    ]

    lines.append("#### Vulnerabilities in binaries (Grype)")
    if findings is None:
        lines.extend(["Grype results not available.", ""])
    elif not findings:
        lines.extend(["✅ No vulnerabilities found in binaries.", ""])
    else:
        counts = Counter(f["severity"] for f in findings)
        lines.extend(
            [
                "| Severity | Count |",
                "|----------|-------|",
                *(f"| {s} | {counts[s]} |" for s in SEVERITY_ORDER if counts[s]),
                "",
            ]
        )
        top = [f for f in findings if f["severity"] in ("Critical", "High")]
        if top:
            lines.extend(
                [
                    "<details>",
                    "<summary>Critical and High details</summary>",
                    "",
                    "| CVE ID | Package | Installed Version | Fixed In | Severity | Location |",
                    "|--------|---------|-------------------|----------|----------|----------|",
                ]
            )
            lines.extend(
                f"| {f['cve']} | {f['package']} | {f['version']} | {f['fixed']} | {f['severity']} | {f['location']} |"
                for f in top[:MAX_ROWS]
            )
            if len(top) > MAX_ROWS:
                lines.append(f"| ... | | | *({len(top) - MAX_ROWS} more)* | | |")
            lines.extend(["", "</details>", ""])

    lines.append("#### Executables not checked for vulnerabilities")
    if unscanned is None:
        lines.extend(["SBOM not available.", ""])
    elif not unscanned:
        lines.extend(["✅ Every executable belongs to a known package.", ""])
    else:
        lines.extend(
            [
                f"{len(unscanned)} executables belong to no known package (e.g. built from source), "
                "so no scanner checks them:",
                "",
                *(f"- `{p}`" for p in unscanned[:MAX_ROWS]),
            ]
        )
        if len(unscanned) > MAX_ROWS:
            lines.append(f"- *({len(unscanned) - MAX_ROWS} more)*")
        lines.append("")
    return "\n".join(lines)


def emit_annotations(image: str, findings: list[dict] | None, unscanned: list[str] | None) -> None:
    for f in [f for f in findings or [] if f["severity"] == "Critical"][:MAX_ANNOTATIONS]:
        print(
            f"::warning title=Binary {f['cve']}::{f['package']} {f['version']} "
            f"(fixed in {f['fixed']}) in {f['location']}"
        )
    if unscanned:
        print(f"::warning title=Unscanned executables in {image}::{len(unscanned)} executables (see job summary)")


def _load(path: str, parser, label: str):
    if not Path(path).is_file():
        print(f"::warning::{label} not found: {path}")
        return None
    return parser(Path(path))


def main() -> None:
    ap = argparse.ArgumentParser(description="Report binaries that Trivy does not inventory")
    ap.add_argument("--grype-results", required=True, help="Grype JSON output (scan of the SBOM)")
    ap.add_argument("--sbom", required=True, help="Syft JSON SBOM")
    ap.add_argument("--image", required=True, help="Image reference (for the summary)")
    ap.add_argument("--github-step-summary", default="", help="Path to GITHUB_STEP_SUMMARY file")
    args = ap.parse_args()

    findings = _load(args.grype_results, parse_grype, "Grype results")
    unscanned = _load(args.sbom, parse_unscanned, "SBOM")
    summary = generate_summary(args.image, findings, unscanned)
    print(summary)
    if args.github_step_summary:
        with open(args.github_step_summary, "a") as f:
            f.write(summary)
    emit_annotations(args.image, findings, unscanned)


if __name__ == "__main__":
    main()
