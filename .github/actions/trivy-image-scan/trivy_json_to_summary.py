#!/usr/bin/env python3
"""
Parse Trivy JSON output, generate a GitHub Actions summary and apply the CVE gate.

Gate policy:
  - Blocking finding: a CRITICAL vulnerability that is either in a language package
    (jar, pip, npm, ...) or in an OS package with a published fix.
    Unfixed CRITICAL OS vulnerabilities and all HIGH vulnerabilities are flagged only.
    Secret findings are not reported here; they are only in the uploaded JSON artifact.
  - policy=warn:  every finding is reported as a warning annotation, exit code 0.
  - policy=block: blocking findings are reported as error annotations and the script
    exits with code 1. A missing results file also blocks (the image could not be verified).
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

# GitHub shows at most 10 annotations of each type per step, so keep the per-CVE list short
MAX_ANNOTATIONS = 10
MAX_DISPLAYED_CVES = 50


def parse_trivy_json(json_path: Path) -> dict:
    """
    Parse Trivy JSON format and extract vulnerability information.

    Trivy JSON structure:
    {
      "Results": [
        {
          "Target": "image_name",
          "Class": "os-pkgs" | "lang-pkgs",
          "Vulnerabilities": [
            {
              "VulnerabilityID": "CVE-2026-12345",
              "PkgName": "perl",
              "PkgPath": "usr/share/app.jar",
              "InstalledVersion": "5.36.0",
              "FixedVersion": "",
              "Severity": "CRITICAL",
              "Title": "..."
            }
          ]
        }
      ]
    }
    """
    with open(json_path) as f:
        data = json.load(f)

    findings = []
    counts = defaultdict(int)

    # Trivy JSON has a "Results" array with vulnerability lists
    for result in data.get("Results", []):
        result_class = result.get("Class", "")
        for vuln in result.get("Vulnerabilities") or []:
            severity = vuln.get("Severity", "UNKNOWN")
            counts[severity] += 1
            fixed = vuln.get("FixedVersion", "")
            is_lang_pkg = result_class == "lang-pkgs"

            findings.append(
                {
                    "cve": vuln.get("VulnerabilityID", "N/A"),
                    "package": vuln.get("PkgName", "N/A"),
                    "version": vuln.get("InstalledVersion", "N/A"),
                    "fixed": fixed or "NotAvailable",
                    "severity": severity,
                    "location": vuln.get("PkgPath") or result.get("Target", ""),
                    "blocking": severity == "CRITICAL" and (is_lang_pkg or bool(fixed)),
                }
            )

    return {"counts": dict(counts), "findings": findings}


def generate_summary(image_name: str, image_tag: str, data: dict, policy: str, scan_type: str = "image") -> str:
    """Generate ECR-style markdown summary."""
    counts = data["counts"]
    findings = data["findings"]
    blocking = [f for f in findings if f["blocking"]]

    lines = [
        "",
        "---",
        "### Trivy Image Scan Results" if scan_type == "image" else "### Trivy Repository (fs) Scan Results",
        f"**Image:** `{image_name}:{image_tag}`",
        f"**CVE policy:** `{policy}` (blocks on CRITICAL with a fix, or CRITICAL in language packages)",
        "",
    ]

    total = sum(counts.values())
    if total == 0:
        lines.append("✅ No vulnerabilities found.")
        return "\n".join(lines)

    # Severity table
    lines.extend(
        [
            "| Severity | Count |",
            "|----------|-------|",
            f"| CRITICAL | {counts.get('CRITICAL', 0)} |",
            f"| HIGH | {counts.get('HIGH', 0)} |",
            f"| Blocking | {len(blocking)} |",
            "",
        ]
    )

    if blocking:
        alert = "CAUTION" if policy == "block" else "WARNING"
        verdict = "release blocked" if policy == "block" else "would block a release"
        lines.extend(
            [
                f"> [!{alert}]",
                f"> **{len(blocking)} blocking CRITICAL vulnerabilities detected ({verdict})!**",
                "",
            ]
        )
    else:
        lines.extend(["✅ No blocking vulnerabilities.", ""])

    # Add only CRITICAL findings to table (limit for readability)
    critical_findings = [f for f in findings if f["severity"] == "CRITICAL"]
    if critical_findings:
        lines.extend(
            [
                "<details>",
                "<summary>Critical CVE Details</summary>",
                "",
                "| CVE ID | Package | Installed Version | Fixed In | Blocking | Location |",
                "|--------|---------|-------------------|----------|----------|----------|",
            ]
        )
        for f in critical_findings[:MAX_DISPLAYED_CVES]:
            lines.append(
                f"| {f['cve']} | {f['package']} | {f['version']} | {f['fixed']} | "
                f"{'yes' if f['blocking'] else 'no'} | {f['location']} |"
            )

        if len(critical_findings) > MAX_DISPLAYED_CVES:
            remaining = len(critical_findings) - MAX_DISPLAYED_CVES
            lines.append(f"| ... | ... | ... | *({remaining} more)* | | |")

        lines.extend(["", "</details>", ""])

    lines.append("_Full scan results (JSON format) attached as workflow artifact._")
    return "\n".join(lines)


def emit_annotations(image: str, data: dict, policy: str) -> None:
    """Print GitHub workflow commands so findings show up as annotations on the run."""
    findings = data["findings"]
    blocking = [f for f in findings if f["blocking"]]
    level = "error" if policy == "block" else "warning"

    for f in blocking[:MAX_ANNOTATIONS]:
        print(
            f"::{level} title=Blocking {f['cve']}::{f['package']} {f['version']} "
            f"(fixed in {f['fixed']}) in {f['location']}"
        )

    counts = data["counts"]
    if findings:
        print(
            f"::warning title=Trivy findings in {image}::"
            f"CRITICAL={counts.get('CRITICAL', 0)} HIGH={counts.get('HIGH', 0)} "
            f"blocking={len(blocking)} (see job summary)"
        )


def write_outputs(data: dict, *, blocked: bool) -> None:
    """Expose the gate result as step outputs."""
    github_output = os.environ.get("GITHUB_OUTPUT")
    if not github_output:
        return
    counts = data["counts"]
    with open(github_output, "a") as f:
        f.write(f"blocked={'true' if blocked else 'false'}\n")
        f.write(f"blocking-count={sum(1 for x in data['findings'] if x['blocking'])}\n")
        f.write(f"critical-count={counts.get('CRITICAL', 0)}\n")
        f.write(f"high-count={counts.get('HIGH', 0)}\n")


def main():
    parser = argparse.ArgumentParser(description="Parse Trivy JSON output, generate summary and apply the CVE gate")
    parser.add_argument("--results-file", required=True, help="Path to Trivy JSON output")
    parser.add_argument("--image-name", required=True, help="Docker image name")
    parser.add_argument("--image-tag", required=True, help="Docker image tag")
    parser.add_argument("--scan-type", choices=["image", "fs"], default="image", help="Trivy scan type of the results")
    parser.add_argument(
        "--policy",
        choices=["warn", "block"],
        default="warn",
        help="warn: report only; block: exit 1 on blocking findings",
    )
    parser.add_argument(
        "--github-step-summary",
        default="",
        help="Path to GITHUB_STEP_SUMMARY file",
    )
    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        level = "error" if args.policy == "block" else "warning"
        print(f"::{level}::Trivy results file not found: {results_path}")
        write_outputs({"counts": {}, "findings": []}, blocked=args.policy == "block")
        if args.policy == "block":
            raise SystemExit(1)
        return

    data = parse_trivy_json(results_path)
    summary = generate_summary(args.image_name, args.image_tag, data, args.policy, args.scan_type)

    print(summary)

    if args.github_step_summary:
        with open(args.github_step_summary, "a") as f:
            f.write(summary)

    emit_annotations(f"{args.image_name}:{args.image_tag} ({args.scan_type} scan)", data, args.policy)

    has_blocking = any(f["blocking"] for f in data["findings"])
    blocked = args.policy == "block" and has_blocking
    write_outputs(data, blocked=blocked)
    if blocked:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
