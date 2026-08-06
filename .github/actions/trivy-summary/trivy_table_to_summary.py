#!/usr/bin/env python3
"""
Parse Trivy table output and generate ECR-style GitHub Actions summary.
"""

import argparse
import re
from pathlib import Path


def parse_trivy_table(table_path: Path) -> dict:
    """
    Parse Trivy table format and extract vulnerability information.

    Trivy table format:
    ugbio_core (debian 12.9)

    Total: 436 (UNKNOWN: 0, LOW: 169, MEDIUM: 175, HIGH: 87, CRITICAL: 5)

    ┌─────────────────┬────────────────┬──────────┬──────────┬────────────────┬─────────┐
    │     Library     │ Vulnerability  │ Severity │  Status  │    Version     │  Fixed  │
    ├─────────────────┼────────────────┼──────────┼──────────┼────────────────┼─────────┤
    │ perl            │ CVE-2026-57433 │ CRITICAL │  fixed   │ 5.36.0-7+deb12 │         │
    └─────────────────┴────────────────┴──────────┴──────────┴────────────────┴─────────┘
    """
    findings = []

    with open(table_path) as f:
        content = f.read()

    # Extract total counts from summary line
    total_match = re.search(r"Total: (\d+) \((.*?)\)", content)
    counts = {"UNKNOWN": 0, "LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}

    if total_match:
        counts_str = total_match.group(2)
        for part in counts_str.split(", "):
            match = re.match(r"(\w+): (\d+)", part)
            if match:
                counts[match.group(1)] = int(match.group(2))

    # Parse table rows (simplified - match CVE pattern lines)
    # Look for lines with CVE-YYYY-NNNNN pattern
    lines = content.split("\n")
    for line in lines:
        # Match table rows with CVE IDs
        match = re.search(
            r"│\s*([^\s│]+)\s*│\s*(CVE-\d{4}-\d+|GHSA-[\w-]+|TEMP-[\w-]+)\s*│\s*(\w+)\s*│\s*\w+\s*│\s*([^\s│]+)\s*│\s*([^\s│]*)\s*│",
            line,
        )
        if match:
            findings.append(
                {
                    "package": match.group(1).strip(),
                    "cve": match.group(2).strip(),
                    "severity": match.group(3).strip(),
                    "version": match.group(4).strip(),
                    "fixed": match.group(5).strip() or "NotAvailable",
                }
            )

    return {"counts": counts, "findings": findings}


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
            f"| CRITICAL | {counts['CRITICAL']} |",
            f"| HIGH | {counts['HIGH']} |",
            f"| MEDIUM | {counts['MEDIUM']} |",
            f"| LOW | {counts['LOW']} |",
            "",
        ]
    )

    # Alert for CRITICAL/HIGH
    critical_high_count = counts["CRITICAL"] + counts["HIGH"]
    if critical_high_count > 0:
        lines.extend(
            [
                "> [!CAUTION]",
                f"> **{critical_high_count} CRITICAL/HIGH vulnerabilities detected!**",
                "",
                "<details>",
                "<summary>Critical & High CVE Details</summary>",
                "",
                "| CVE ID | Package | Installed Version | Fixed In | Severity |",
                "|--------|---------|-------------------|----------|----------|",
            ]
        )

        # Add CRITICAL and HIGH findings to table
        for f in findings:
            if f["severity"] in ["CRITICAL", "HIGH"]:
                lines.append(f"| {f['cve']} | {f['package']} | {f['version']} | " f"{f['fixed']} | {f['severity']} |")

        lines.extend(["", "</details>", ""])

    lines.append("_Full scan results (table format) attached as workflow artifact._")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Parse Trivy table output and generate summary")
    parser.add_argument("--table-file", required=True, help="Path to Trivy table output")
    parser.add_argument("--image-name", required=True, help="Docker image name")
    parser.add_argument("--image-tag", required=True, help="Docker image tag")
    parser.add_argument(
        "--github-step-summary",
        default="",
        help="Path to GITHUB_STEP_SUMMARY file",
    )
    args = parser.parse_args()

    table_path = Path(args.table_file)
    if not table_path.exists():
        print(f"::warning::Trivy table file not found: {table_path}")
        return

    data = parse_trivy_table(table_path)
    summary = generate_summary(args.image_name, args.image_tag, data)

    print(summary)

    if args.github_step_summary:
        with open(args.github_step_summary, "a") as f:
            f.write(summary)


if __name__ == "__main__":
    main()
