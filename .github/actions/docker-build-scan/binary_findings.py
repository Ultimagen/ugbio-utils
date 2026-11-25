#!/usr/bin/env python3
"""
Report software that Trivy does not inventory: binaries installed outside a package manager.

Two sources:
  - Grype matches on Syft "binary" packages, i.e. versions detected from the executable itself
    (e.g. CPython built into /usr/local by the official python images). Matches on OS and
    language packages are skipped: Trivy already reports those. With policy=block, a Critical
    match that has a fix, or missing Grype results, fails (exit 1).
  - Executables Syft could not attribute to any package (source-built tools such as samtools).
    Nothing checks these for vulnerabilities, so they are listed for review (never blocking).
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
        fixed = fix.get("versions") or []
        severity = vuln.get("severity", "Unknown")
        findings.append(
            {
                "cve": vuln.get("id", "N/A"),
                "package": artifact.get("name", "N/A"),
                "version": artifact.get("version", "N/A"),
                "severity": severity,
                "fixed": ", ".join(fixed) or "NotAvailable",
                "blocking": severity == "Critical" and bool(fixed),
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


def generate_summary(image: str, findings: list[dict] | None, unscanned: list[str] | None, policy: str) -> str:
    lines = [
        "",
        "---",
        "### Binaries not covered by Trivy",
        f"**Image:** `{image}`",
        "",
    ]

    lines.append("#### Vulnerabilities in binaries (Grype)")
    if findings is None:
        verdict = "release blocked" if policy == "block" else "would block a release"
        lines.extend(["> [!CAUTION]", f"> **Grype results not available ({verdict}).**", ""])
    elif not findings:
        lines.extend(["✅ No vulnerabilities found in binaries.", ""])
    else:
        counts = Counter(f["severity"] for f in findings)
        blocking = sum(f["blocking"] for f in findings)
        lines.extend(
            [
                "| Severity | Count |",
                "|----------|-------|",
                *(f"| {s} | {counts[s]} |" for s in SEVERITY_ORDER if counts[s]),
                f"| Blocking | {blocking} |",
                "",
            ]
        )
        if blocking:
            verdict = "release blocked" if policy == "block" else "would block a release"
            lines.extend(
                ["> [!CAUTION]", f"> **{blocking} Critical vulnerabilities with a fix in binaries ({verdict})!**", ""]
            )
        top = [f for f in findings if f["severity"] in ("Critical", "High")]
        if top:
            lines.extend(
                [
                    "<details>",
                    "<summary>Critical and High details</summary>",
                    "",
                    "| CVE ID | Package | Installed Version | Fixed In | Severity | Blocking | Location |",
                    "|--------|---------|-------------------|----------|----------|----------|----------|",
                ]
            )
            lines.extend(
                f"| {f['cve']} | {f['package']} | {f['version']} | {f['fixed']} | {f['severity']} | "
                f"{'yes' if f['blocking'] else 'no'} | {f['location']} |"
                for f in top[:MAX_ROWS]
            )
            if len(top) > MAX_ROWS:
                lines.append(f"| ... | | | *({len(top) - MAX_ROWS} more)* | | | |")
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


def emit_annotations(image: str, findings: list[dict] | None, unscanned: list[str] | None, policy: str) -> None:
    for f in [f for f in findings or [] if f["severity"] == "Critical"][:MAX_ANNOTATIONS]:
        level = "error" if policy == "block" and f["blocking"] else "warning"
        print(
            f"::{level} title=Binary {f['cve']}::{f['package']} {f['version']} "
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
    ap.add_argument(
        "--policy",
        choices=["warn", "block"],
        default="warn",
        help="warn: report only; block: exit 1 on a Critical binary vulnerability with a fix or missing Grype results",
    )
    ap.add_argument("--github-step-summary", default="", help="Path to GITHUB_STEP_SUMMARY file")
    args = ap.parse_args()

    findings = _load(args.grype_results, parse_grype, "Grype results")
    unscanned = _load(args.sbom, parse_unscanned, "SBOM")
    summary = generate_summary(args.image, findings, unscanned, args.policy)
    print(summary)
    if args.github_step_summary:
        with open(args.github_step_summary, "a") as f:
            f.write(summary)
    emit_annotations(args.image, findings, unscanned, args.policy)

    blocked = args.policy == "block" and (findings is None or any(f["blocking"] for f in findings))
    if blocked:
        print("::error::Binaries check failed: Critical vulnerabilities with a fix, or no Grype results")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
