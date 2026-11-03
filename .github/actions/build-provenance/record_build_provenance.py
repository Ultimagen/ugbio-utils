#!/usr/bin/env python3
"""
Record which ugbio-utils commit runs the shared docker workflow, and which commit of the built
repository is checked out, as GitHub Actions step outputs.

Callers use the shared workflow by branch (docker-build-push.yml@main), so ugbio-utils can change
between builds; the values end up in the image labels and the run summary.

  - ugbio_utils_ref / ugbio_utils_sha: the job's OIDC token names the workflow file that runs the
    job and its commit (job_workflow_ref / job_workflow_sha). Callers already grant id-token: write.
    Fallback: the run's referenced_workflows, through the GitHub API.
  - source_sha: `git rev-parse HEAD` in the working directory.

Recording provenance never fails the build: a value that cannot be found is written as "unknown".
"""

import argparse
import base64
import json
import os
import shutil
import subprocess
import urllib.parse
import urllib.request

UNKNOWN = "unknown"
OIDC_AUDIENCE = "ugbio-utils-provenance"


def workflow_from_oidc_token() -> tuple[str, str]:
    """(job_workflow_ref, job_workflow_sha) from the job's OIDC token, or empty strings."""
    url = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    request_token = os.environ.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    if not url or not request_token:
        return "", ""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        print(f"::warning::Not reading the OIDC token from a non-https URL ({parsed.scheme})")
        return "", ""
    separator = "&" if parsed.query else "?"
    request = urllib.request.Request(  # noqa: S310 - https only, checked above
        f"{url}{separator}audience={OIDC_AUDIENCE}", headers={"Authorization": f"bearer {request_token}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - https only, checked above
            token = json.load(response)["value"]
        payload = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except Exception as error:  # noqa: BLE001 - provenance must not fail the build
        print(f"::warning::Could not read the workflow commit from the OIDC token: {error}")
        return "", ""
    return claims.get("job_workflow_ref", ""), claims.get("job_workflow_sha", "")


def workflow_from_run(workflow_path: str) -> tuple[str, str]:
    """(path@ref, sha) of workflow_path among the run's referenced_workflows, or empty strings."""
    repository = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    gh = shutil.which("gh")
    if not repository or not run_id or not gh:
        return "", ""
    try:
        result = subprocess.run(
            [gh, "api", f"repos/{repository}/actions/runs/{run_id}"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        for workflow in json.loads(result.stdout).get("referenced_workflows", []):
            if workflow.get("path", "").startswith(f"{workflow_path}@"):
                return workflow["path"], workflow.get("sha", "")
    except Exception as error:  # noqa: BLE001 - provenance must not fail the build
        print(f"::warning::Could not read the workflow commit from the run: {error}")
    return "", ""


def checked_out_commit() -> str:
    git = shutil.which("git")
    if not git:
        print("::warning::git not found; cannot read the checked-out commit")
        return ""
    try:
        result = subprocess.run([git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception as error:  # noqa: BLE001 - provenance must not fail the build
        print(f"::warning::Could not read the checked-out commit: {error}")
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--workflow-path",
        default="Ultimagen/ugbio-utils/.github/workflows/docker-build-push.yml",
        help="owner/repo/path of the shared workflow, for the referenced_workflows fallback",
    )
    args = parser.parse_args()

    ref, sha = workflow_from_oidc_token()
    if not sha:
        ref, sha = workflow_from_run(args.workflow_path)
    outputs = {
        "ugbio_utils_ref": ref or UNKNOWN,
        "ugbio_utils_sha": sha or UNKNOWN,
        "source_sha": checked_out_commit() or UNKNOWN,
    }

    output_file = os.environ.get("GITHUB_OUTPUT")
    if output_file:
        with open(output_file, "a", encoding="utf-8") as handle:
            for name, value in outputs.items():
                handle.write(f"{name}={value}\n")
    for name, value in outputs.items():
        print(f"{name}={value}")


if __name__ == "__main__":
    main()
