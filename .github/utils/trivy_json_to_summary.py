#!/usr/bin/env python3
"""
Parse Trivy JSON output and generate ECR-style GitHub Actions summary.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_trivy_json(json_path: Path) -> dict:
    """
    Parse Trivy JSON format and extract vulnerability information.

    Trivy JSON structure:
    {
      "Results": [
        {
          "Target": "image_name",
          "Vulnerabilities": [
            {
              "VulnerabilityID": "CVE-2026-12345",
              "PkgName": "perl",
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
        for vuln in result.get("Vulnerabilities", []):
            severity = vuln.get("Severity", "UNKNOWN")
            counts[severity] += 1

            findings.append(
                {
                    "cve": vuln.get("VulnerabilityID", "N/A"),
                    "package": vuln.get("PkgName", "N/A"),
                    "version": vuln.get("InstalledVersion", "N/A"),
                    "fixed": vuln.get("FixedVersion", "") or "NotAvailable",
                    "severity": severity,
                }
            )

    return {"counts": dict(counts), "findings": findings}


def generate_summary(image_name: str, image_tag: str, data: dict) -> str:
    """Generate ECR-style markdown summary."""
    counts = data["counts"]
    findings = data["findings"]

    lines = [
        "",
        "---",
        "### Trivy Image Scan Results",
        f"**Image:** `{image_name}:{image_tag}`",
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
            "",
        ]
    )

    # Alert for CRITICAL
    critical_count = counts.get("CRITICAL", 0)
    if critical_count > 0:
        lines.extend(
            [
                "> [!CAUTION]",
                f"> **{critical_count} CRITICAL vulnerabilities detected!**",
                "",
                "<details>",
                "<summary>Critical CVE Details</summary>",
                "",
                "| CVE ID | Package | Installed Version | Fixed In |",
                "|--------|---------|-------------------|----------|",
            ]
        )

        # Add only CRITICAL findings to table (limit for readability)
        max_displayed_cves = 50
        critical_findings = [f for f in findings if f["severity"] == "CRITICAL"]
        for f in critical_findings[:max_displayed_cves]:
            lines.append(f"| {f['cve']} | {f['package']} | {f['version']} | {f['fixed']} |")

        if len(critical_findings) > max_displayed_cves:
            remaining = len(critical_findings) - max_displayed_cves
            lines.append(f"| ... | ... | ... | *({remaining} more)* |")

        lines.extend(["", "</details>", ""])

    lines.append("_Full scan results (JSON format) attached as workflow artifact._")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Parse Trivy JSON output and generate summary")
    parser.add_argument("--results-file", required=True, help="Path to Trivy JSON output")
    parser.add_argument("--image-name", required=True, help="Docker image name")
    parser.add_argument("--image-tag", required=True, help="Docker image tag")
    parser.add_argument(
        "--github-step-summary",
        default="",
        help="Path to GITHUB_STEP_SUMMARY file",
    )
    args = parser.parse_args()

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(f"::warning::Trivy results file not found: {results_path}")
        return

    data = parse_trivy_json(results_path)
    summary = generate_summary(args.image_name, args.image_tag, data)

    print(summary)

    if args.github_step_summary:
        with open(args.github_step_summary, "a") as f:
            f.write(summary)


if __name__ == "__main__":
    main()
