import configparser
import logging
import os
import shutil
import subprocess

import pysam

logger = logging.getLogger(__name__)


def _get_aws_cli_path() -> str:
    aws_cli = shutil.which("aws")
    if not aws_cli:
        raise RuntimeError("AWS CLI not found in PATH")
    return aws_cli


def _parse_aws_config(config_path: str = "~/.aws/config") -> dict[str, str]:
    cfg_path = os.path.expanduser(config_path)
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"AWS config file not found at {cfg_path}")
    cfg = configparser.ConfigParser()
    try:
        cfg.read(cfg_path)
    except configparser.Error as e:
        raise RuntimeError(f"Error parsing AWS config file {cfg_path}: {e}") from e

    acct2prof: dict[str, str] = {}
    for section in cfg.sections():
        if not section.startswith("profile "):
            continue
        prof = section.removeprefix("profile ").strip()
        if cfg.has_option(section, "sso_account_id"):
            acct2prof[cfg.get(section, "sso_account_id").strip()] = prof
    return acct2prof


def _deduce_profile(cram_file: str, acct2prof: dict[str, str]) -> str:
    matches = [p for acct, p in acct2prof.items() if acct in cram_file]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple AWS accounts {matches} in path; cannot deduce profile.")
    raise ValueError(f"No SSO account IDs {list(acct2prof.keys())} found in {cram_file}; please specify profile.")


def _export_sso_env(profile: str) -> None:
    aws_cli = _get_aws_cli_path()
    try:
        out = (
            subprocess.check_output(
                [aws_cli, "configure", "export-credentials", "--profile", profile, "--format", "env-no-export"]
            )
            .decode()
            .splitlines()
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to export credentials for profile '{profile}'") from e

    for line in out:
        k, v = line.split("=", 1)
        os.environ[k] = v


def read_cram_from_s3(
    cram_file: str,
    profile: str | None = None,
    ref_fasta: str | None = None,
) -> pysam.AlignmentFile:
    """
    Read a CRAM file from S3 using the AWS CLI and pysam.
    This function attempts to deduce the AWS CLI profile from the S3 URI
    if not provided, using the ~/.aws/config file.
    It requires the AWS CLI to be installed and configured with SSO credentials.

    :param cram_file: S3 URI, e.g. "s3://…/file.cram"
    :param profile: AWS CLI profile; if None, attempt to deduce from ~/.aws/config
    :param ref_fasta: local reference FASTA (required)
    :returns: an open pysam.AlignmentFile (caller must close it)
    """
    if profile is None:
        try:
            acct2prof = _parse_aws_config()
            profile = _deduce_profile(cram_file, acct2prof)
        except Exception as e:
            logger.debug(f"Profile deduction failed ({e}); proceeding without exporting SSO credentials.")

    if profile:
        _export_sso_env(profile)

    os.environ.setdefault("AWS_DEFAULT_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    if not cram_file.endswith(".cram"):
        raise ValueError(f"CRAM file {cram_file} must end with .cram")
    try:
        sam = pysam.AlignmentFile(cram_file, "rc", reference_filename=ref_fasta)
    except OSError as e:
        if "Permission denied" in str(e):
            raise RuntimeError(
                f"Permission denied reading {cram_file}. " f"Did you run `aws sso login --profile {profile}`?"
            ) from e
        raise
    return sam


def read_bam_from_s3(
    bam_file: str,
    profile: str | None = None,
) -> pysam.AlignmentFile:
    """
    Read a CRAM file from S3 using the AWS CLI and pysam.
    This function attempts to deduce the AWS CLI profile from the S3 URI
    if not provided, using the ~/.aws/config file.
    It requires the AWS CLI to be installed and configured with SSO credentials.

    :param bam_file: S3 URI, e.g. "s3://…/file.bam"
    :param profile: AWS CLI profile; if None, attempt to deduce from ~/.aws/config
    :returns: an open pysam.AlignmentFile (caller must close it)
    """
    if profile is None:
        try:
            acct2prof = _parse_aws_config()
            profile = _deduce_profile(bam_file, acct2prof)
        except Exception as e:
            logger.debug(f"Profile deduction failed ({e}); proceeding without exporting SSO credentials.")

    if profile:
        _export_sso_env(profile)

    os.environ.setdefault("AWS_DEFAULT_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    if not bam_file.endswith(".bam"):
        raise ValueError(f"BAM file {bam_file} must end with .bam")
    try:
        sam = pysam.AlignmentFile(bam_file, "r")
    except OSError as e:
        if "Permission denied" in str(e):
            raise RuntimeError(
                f"Permission denied reading {bam_file}. " f"Did you run `aws sso login --profile {profile}`?"
            ) from e
        raise
    return sam


def read_vcf_from_s3(
    vcf_file: str,
    profile: str | None = None,
) -> pysam.VariantFile:
    """
    Read a VCF file from S3 using the AWS CLI and pysam.
    This function attempts to deduce the AWS CLI profile from the S3 URI
    if not provided, using the ~/.aws/config file.
    It requires the AWS CLI to be installed and configured with SSO credentials.

    :param vcf_file: S3 URI, e.g. "s3://…/file.vcf.gz or .vcf"
    :param profile: AWS CLI profile; if None, attempt to deduce from ~/.aws/config
    :returns: an open pysam.AlignmentFile (caller must close it)
    """
    if profile is None:
        try:
            acct2prof = _parse_aws_config()
            profile = _deduce_profile(vcf_file, acct2prof)
        except Exception as e:
            logger.debug(f"Profile deduction failed ({e}); proceeding without exporting SSO credentials.")

    if profile:
        _export_sso_env(profile)

    os.environ.setdefault("AWS_DEFAULT_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    if not vcf_file.endswith((".vcf", ".vcf.gz")):
        raise ValueError(f"VCF file {vcf_file} must end with .vcf or .vcf.gz")
    try:
        vcf = pysam.VariantFile(vcf_file, "r")
    except OSError as e:
        if "Permission denied" in str(e):
            raise RuntimeError(
                f"Permission denied reading {vcf_file}. " f"Did you run `aws sso login --profile {profile}`?"
            ) from e
        raise
    return vcf
