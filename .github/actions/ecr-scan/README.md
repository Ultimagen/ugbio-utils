# ECR Enhanced Scan Results Action

This composite action retrieves AWS Inspector2 scan results for a Docker image in Amazon ECR and generates a GitHub Actions job summary with vulnerability details.

## Features

- Polls AWS Inspector2 for Enhanced Scan findings
- Generates formatted job summary with vulnerability counts
- Exports detailed findings as JSON artifact
- Supports CRITICAL, HIGH, MEDIUM, and LOW severity levels
- Graceful handling when Enhanced Scanning is not enabled

## Usage

### In a workflow

```yaml
- name: Check ECR scan results
  id: ecr-scan
  continue-on-error: true
  uses: Ultimagen/ugbio-utils/.github/actions/ecr-scan@main
  with:
    repository: ugbio_core
    tag: v1.0.0
    output-dir: .

- name: Upload scan findings
  if: steps.ecr-scan.outputs.scan_status == 'complete'
  uses: actions/upload-artifact@v4
  with:
    name: ecr-scan-findings
    path: ${{ steps.ecr-scan.outputs.findings_file }}
```

### Version pinning

For production workflows, pin to a specific commit or tag:

```yaml
uses: Ultimagen/ugbio-utils/.github/actions/ecr-scan@v1.0.0
# or
uses: Ultimagen/ugbio-utils/.github/actions/ecr-scan@abc1234
```

## Inputs

| Input | Description | Required | Default |
|-------|-------------|----------|---------|
| `repository` | ECR repository name (e.g., `ugbio_core`) | Yes | - |
| `tag` | Image tag (e.g., `v1.0.0`, `main_abc1234`) | Yes | - |
| `output-dir` | Directory to write findings JSON | No | `.` |

## Outputs

| Output | Description | Example |
|--------|-------------|---------|
| `scan_status` | Status: `complete`, `skipped`, or `inspector_unavailable` | `complete` |
| `findings_file` | Path to findings JSON file | `ecr-scan-findings-ugbio_core-v1.0.0.json` |
| `critical_count` | Number of CRITICAL vulnerabilities | `2` |
| `high_count` | Number of HIGH vulnerabilities | `5` |
| `medium_count` | Number of MEDIUM vulnerabilities | `10` |
| `total_count` | Total vulnerability count | `17` |

## AWS Permissions Required

The workflow must have the following AWS IAM permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:DescribeImages",
        "inspector2:ListFindings"
      ],
      "Resource": "*"
    }
  ]
}
```

Typically configured via OIDC with:

```yaml
permissions:
  id-token: write
  contents: read

- name: Configure AWS credentials
  uses: aws-actions/configure-aws-credentials@v3
  with:
    role-to-assume: arn:aws:iam::ACCOUNT_ID:role/github-actions-ecr
    aws-region: us-east-1
```

## Polling Behavior

The action waits for Enhanced Scan results using this strategy:

1. **Initial delay**: 30 seconds (allows scan to start processing)
2. **Poll interval**: 15 seconds
3. **Max attempts**: 12 (total ~3 minutes)
4. **Timeout**: Returns findings even if empty (clean image)

## Error Handling

The action uses `continue-on-error: true` in the common workflow to ensure Docker builds don't fail if:

- Image digest cannot be determined
- Inspector2 API is unavailable
- Enhanced Scanning is not enabled for the repository
- Poll timeout occurs before findings appear

## Job Summary Format

The action appends a formatted summary to `GITHUB_STEP_SUMMARY`:

```markdown
---
### ECR Enhanced Scan Results
**Image:** `ugbio_core:v1.0.0`

| Severity | Count |
|----------|-------|
| CRITICAL | 2 |
| HIGH | 5 |
| MEDIUM | 10 |

> [!CAUTION]
> **2 CRITICAL vulnerabilities detected!**

<details>
<summary>Critical CVE Details</summary>

| CVE ID | Package | Installed Version | Fixed In | Score |
|--------|---------|-------------------|----------|-------|
| CVE-2024-1234 | openssl | 1.1.1 | 1.1.1n | 9.8 |
| CVE-2024-5678 | curl | 7.68.0 | 7.81.0 | 9.1 |

</details>

_Full scan results attached as workflow artifact._
```

## Local Testing

You can test the scan script locally:

```bash
cd .github/actions/ecr-scan

python3 ecr_scan_results.py \
  --repository ugbio_core \
  --tag test_abc1234 \
  --output-dir /tmp \
  --github-output /tmp/github-output.txt \
  --github-step-summary /tmp/summary.md

# View outputs
cat /tmp/github-output.txt
cat /tmp/summary.md
cat /tmp/ecr-scan-findings-ugbio_core-test_abc1234.json
```

## Maintenance

The scan script is located at [`.github/actions/ecr-scan/ecr_scan_results.py`](./ ecr_scan_results.py).

To update:

1. Modify the Python script
2. Test locally or in a branch
3. Merge to `main` (all workflows using `@main` will pick up the change)
4. Tag a new version if using semver pinning

## Related

- Common Docker build workflow: [`.github/workflows/docker-build-push.yml`](../../workflows/docker-build-push.yml)
- AWS Inspector2 documentation: https://docs.aws.amazon.com/inspector/latest/user/what-is-inspector.html
