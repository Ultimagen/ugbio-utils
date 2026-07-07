#!/usr/bin/env python3
"""
Retrieve ECR Enhanced Scan (Inspector2) results for a Docker image
and generate a GitHub Actions job summary with CVE alerts.

Usage:
    python ecr_scan_results.py \
        --repository ugbio_filtering \
        --tag test_abc1234 \
        --output-dir . \
        --github-output "$GITHUB_OUTPUT" \
        --github-step-summary "$GITHUB_STEP_SUMMARY"
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

MAX_ATTEMPTS = 12
POLL_INTERVAL_SECONDS = 15
INITIAL_DELAY_SECONDS = 30


def aws_cli(args: list[str]) -> dict | None:
    """Run an AWS CLI command and return parsed JSON output, or None on failure."""
    cmd = ["aws"] + args + ["--output", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def get_image_digest(repository: str, tag: str) -> str | None:
    """Get the image digest from ECR."""
    data = aws_cli(
        [
            "ecr",
            "describe-images",
            "--repository-name",
            repository,
            "--image-ids",
            f"imageTag={tag}",
        ]
    )
    if not data:
        return None
    details = data.get("imageDetails", [])
    if not details:
        return None
    return details[0].get("imageDigest")


def poll_inspector_findings(repository: str, digest: str) -> dict | None:
    """Poll Inspector2 until findings appear or timeout. Returns findings data or None."""
    print(f"Waiting {INITIAL_DELAY_SECONDS}s for Enhanced Scan to begin processing...")
    time.sleep(INITIAL_DELAY_SECONDS)

    max_wait = MAX_ATTEMPTS * POLL_INTERVAL_SECONDS
    print(f"Polling Inspector2 for findings (max {max_wait}s)...")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        findings_data = get_inspector_findings(repository, digest)
        if findings_data is None:
            print(f"  Attempt {attempt}/{MAX_ATTEMPTS}: Inspector2 API unavailable")
            return None

        count = len(findings_data.get("findings", []))
        print(f"  Attempt {attempt}/{MAX_ATTEMPTS}: {count} findings")

        if count > 0:
            return findings_data

        time.sleep(POLL_INTERVAL_SECONDS)

    # Final attempt — return whatever we have (may be 0 findings = clean image)
    return findings_data


def get_inspector_findings(repository: str, digest: str) -> dict | None:
    """Retrieve findings from Inspector2 filtered by image."""
    filter_criteria = json.dumps(
        {
            "ecrImageHash": [{"comparison": "EQUALS", "value": digest}],
            "ecrImageRepositoryName": [{"comparison": "EQUALS", "value": repository}],
        }
    )
    return aws_cli(
        [
            "inspector2",
            "list-findings",
            "--filter-criteria",
            filter_criteria,
        ]
    )


def count_by_severity(findings: list[dict]) -> dict[str, int]:
    """Count findings grouped by severity."""
    counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for f in findings:
        sev = f.get("severity", "UNKNOWN")
        counts[sev] = counts.get(sev, 0) + 1
    return counts


def generate_summary(image_name: str, image_tag: str, findings: list[dict], counts: dict[str, int]) -> str:
    """Generate markdown summary for GITHUB_STEP_SUMMARY."""
    lines = [
        "",
        "---",
        "### ECR Enhanced Scan Results",
        f"**Image:** `{image_name}:{image_tag}`",
        "",
    ]

    total = sum(counts.values())
    if total == 0:
        lines.append("No vulnerabilities found.")
        return "\n".join(lines)

    # Severity table
    lines.extend(
        [
            "| Severity | Count |",
            "|----------|-------|",
            f"| CRITICAL | {counts.get('CRITICAL', 0)} |",
            f"| HIGH | {counts.get('HIGH', 0)} |",
            f"| MEDIUM | {counts.get('MEDIUM', 0)} |",
            "",
        ]
    )

    # Alert for CRITICALs
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
                "| CVE ID | Package | Installed Version | Fixed In | Score |",
                "|--------|---------|-------------------|----------|-------|",
            ]
        )

        for f in findings:
            if f.get("severity") != "CRITICAL":
                continue
            vuln = f.get("packageVulnerabilityDetails", {})
            cve_id = vuln.get("vulnerabilityId", "N/A")
            pkgs = vuln.get("vulnerablePackages", [{}])
            pkg_name = pkgs[0].get("name", "N/A") if pkgs else "N/A"
            pkg_version = pkgs[0].get("version", "N/A") if pkgs else "N/A"
            fixed_in = pkgs[0].get("fixedInVersion", "No fix") if pkgs else "No fix"
            cvss_list = vuln.get("cvss", [{}])
            score = cvss_list[0].get("baseScore", "N/A") if cvss_list else "N/A"
            lines.append(f"| {cve_id} | {pkg_name} | {pkg_version} | {fixed_in} | {score} |")

        lines.extend(["", "</details>"])

    lines.extend(["", "_Full scan results attached as workflow artifact._"])
    return "\n".join(lines)


def write_github_output(output_file: str, key: str, value: str):
    """Append a key=value pair to GITHUB_OUTPUT."""
    with open(output_file, "a") as f:
        f.write(f"{key}={value}\n")


def main():
    parser = argparse.ArgumentParser(description="Retrieve ECR scan results and generate summary")
    parser.add_argument("--repository", required=True, help="ECR repository name")
    parser.add_argument("--tag", required=True, help="Image tag")
    parser.add_argument("--output-dir", default=".", help="Directory to write findings JSON")
    parser.add_argument(
        "--github-output",
        default=os.environ.get("GITHUB_OUTPUT", ""),
        help="Path to GITHUB_OUTPUT file",
    )
    parser.add_argument(
        "--github-step-summary",
        default=os.environ.get("GITHUB_STEP_SUMMARY", ""),
        help="Path to GITHUB_STEP_SUMMARY file",
    )
    args = parser.parse_args()

    def set_output(key, value):
        if args.github_output:
            write_github_output(args.github_output, key, value)

    # Get image digest
    print(f"Looking up image: {args.repository}:{args.tag}")
    digest = get_image_digest(args.repository, args.tag)
    if not digest:
        print("::warning::Could not determine image digest. Scan check skipped.")
        set_output("scan_status", "skipped")
        return

    print(f"Image digest: {digest}")

    # Poll Inspector2 for findings (Enhanced Scanning has no scan-status field)
    findings_data = poll_inspector_findings(args.repository, digest)
    if findings_data is None:
        print("::warning::Could not retrieve Inspector2 findings (Enhanced Scanning may not be enabled).")
        set_output("scan_status", "inspector_unavailable")
        return

    findings = findings_data.get("findings", [])

    # Save findings to file
    findings_file = Path(args.output_dir) / f"ecr-scan-findings-{args.repository}-{args.tag}.json"
    findings_file.write_text(json.dumps(findings_data, indent=2))
    print(f"Findings saved to: {findings_file}")

    # Compute counts
    counts = count_by_severity(findings)
    total = sum(counts.values())
    critical = counts.get("CRITICAL", 0)

    print(f"Scan complete: {total} total ({critical} critical, {counts.get('HIGH', 0)} high)")

    # Write outputs
    set_output("scan_status", "complete")
    set_output("findings_file", str(findings_file))
    set_output("critical_count", str(critical))
    set_output("high_count", str(counts.get("HIGH", 0)))
    set_output("medium_count", str(counts.get("MEDIUM", 0)))
    set_output("total_count", str(total))

    # Write summary
    if args.github_step_summary:
        summary = generate_summary(args.repository, args.tag, findings, counts)
        with open(args.github_step_summary, "a") as f:
            f.write(summary)


if __name__ == "__main__":
    main()
