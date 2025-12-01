import configparser
import logging
import os
import shutil
import subprocess
from collections.abc import Callable
from typing import Any

import pysam

try:
    import certifi
except ImportError:
    certifi = None

logger = logging.getLogger(__name__)

# Constants
DEFAULT_AWS_REGION = "us-east-1"
SUPPORTED_CRAM_EXTENSIONS = (".cram",)
SUPPORTED_BAM_EXTENSIONS = (".bam",)
SUPPORTED_VCF_EXTENSIONS = (".vcf", ".vcf.gz")


def _get_aws_cli_path() -> str:
    """Get the path to the AWS CLI executable."""
    aws_cli = shutil.which("aws")
    if not aws_cli:
        raise RuntimeError("AWS CLI not found in PATH")
    return aws_cli


def _setup_ssl_certificates() -> None:
    """
    Set up SSL certificate bundle for libcurl (used by pysam/htslib for S3 access).

    Sets CURL_CA_BUNDLE or SSL_CERT_FILE environment variable if not already set.
    Tries certifi package first, then common system locations.
    """
    # If already set, don't override
    if os.environ.get("CURL_CA_BUNDLE") or os.environ.get("SSL_CERT_FILE"):
        return

    # Try certifi package first (most reliable)
    if certifi:
        cert_path = certifi.where()
        if os.path.isfile(cert_path):
            os.environ["CURL_CA_BUNDLE"] = cert_path
            logger.debug(f"Set CURL_CA_BUNDLE to {cert_path} (from certifi)")
            return

    # Try common system locations
    common_paths = [
        "/etc/ssl/certs/ca-certificates.crt",  # Debian/Ubuntu
        "/etc/pki/tls/certs/ca-bundle.crt",  # Red Hat/CentOS
        "/etc/ssl/cert.pem",  # Some systems
        "/usr/local/etc/openssl/cert.pem",  # macOS (Homebrew)
    ]

    for cert_path in common_paths:
        if os.path.isfile(cert_path):
            os.environ["CURL_CA_BUNDLE"] = cert_path
            logger.debug(f"Set CURL_CA_BUNDLE to {cert_path} (system location)")
            return

    logger.warning(
        "Could not find SSL certificate bundle. S3 access may fail with SSL errors. "
        "Consider installing certifi package or setting CURL_CA_BUNDLE environment variable."
    )


def _parse_aws_config(config_path: str = "~/.aws/config") -> dict[str, str]:
    """Parse AWS config file and return mapping of account IDs to profiles."""
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


def _deduce_profile(file_path: str, acct2prof: dict[str, str]) -> str:
    """Deduce AWS profile from file path using account ID mapping."""
    matches = [p for acct, p in acct2prof.items() if acct in file_path]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Multiple AWS accounts {matches} in path; cannot deduce profile.")
    raise ValueError(f"No SSO account IDs {list(acct2prof.keys())} found in {file_path}; please specify profile.")


def _export_sso_env(profile: str) -> None:
    """Export SSO credentials to environment variables."""
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


def _setup_aws_credentials(file_path: str, profile: str | None = None) -> str | None:
    """
    Set up AWS credentials for S3 access.

    :param file_path: S3 URI of the file
    :param profile: AWS CLI profile; if None, attempt to deduce from ~/.aws/config
    :returns: The profile used (for error messages), or None if no profile was set up
    """
    # Set up SSL certificates first (needed for libcurl S3 access)
    _setup_ssl_certificates()

    if profile is None:
        try:
            acct2prof = _parse_aws_config()
            profile = _deduce_profile(file_path, acct2prof)
        except Exception as e:
            logger.debug(f"Profile deduction failed ({e}); proceeding without exporting SSO credentials.")
            return None

    if profile:
        _export_sso_env(profile)
        os.environ.setdefault("AWS_DEFAULT_REGION", DEFAULT_AWS_REGION)

    return profile


def _validate_file_extension(file_path: str, supported_extensions: tuple[str, ...]) -> None:
    """Validate that the file has a supported extension."""
    if not any(file_path.endswith(ext) for ext in supported_extensions):
        ext_list = ", ".join(supported_extensions)
        raise ValueError(f"File {file_path} must end with one of: {ext_list}")


def _read_file_from_s3(
    file_path: str,
    profile: str | None,
    supported_extensions: tuple[str, ...],
    file_reader_func: Callable[..., Any],
    *args,
    **kwargs,
) -> Any:
    """
    Generic function to read files from S3 with common setup and error handling.

    :param file_path: S3 URI of the file
    :param profile: AWS CLI profile
    :param supported_extensions: Tuple of supported file extensions
    :param file_reader_func: Function to call for reading the file (e.g., pysam.AlignmentFile)
    :param args: Additional positional arguments for file_reader_func
    :param kwargs: Additional keyword arguments for file_reader_func
    :returns: Result of file_reader_func
    """
    _validate_file_extension(file_path, supported_extensions)
    used_profile = _setup_aws_credentials(file_path, profile)

    try:
        return file_reader_func(file_path, *args, **kwargs)
    except OSError as e:
        error_str = str(e)
        if "Permission denied" in error_str:
            profile_msg = f" Did you run `aws sso login --profile {used_profile}`?" if used_profile else ""
            raise RuntimeError(f"Permission denied reading {file_path}.{profile_msg}") from e
        if "SSL" in error_str or "CA cert" in error_str or "libcurl" in error_str.lower():
            curl_ca_bundle = os.environ.get("CURL_CA_BUNDLE", "not set")
            if os.environ.get("CURL_CA_BUNDLE"):
                cert_msg = f" SSL certificate issue detected. CURL_CA_BUNDLE is set to {curl_ca_bundle}."
            else:
                cert_msg = (
                    " SSL certificate bundle not found. Consider installing certifi package "
                    "or setting CURL_CA_BUNDLE."
                )
            raise RuntimeError(f"SSL certificate error reading {file_path}.{cert_msg}") from e
        raise


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
    :param ref_fasta: local reference FASTA (required for CRAM files)
    :returns: an open pysam.AlignmentFile (caller must close it)
    """
    return _read_file_from_s3(
        cram_file, profile, SUPPORTED_CRAM_EXTENSIONS, pysam.AlignmentFile, "rc", reference_filename=ref_fasta
    )


def read_bam_from_s3(
    bam_file: str,
    profile: str | None = None,
) -> pysam.AlignmentFile:
    """
    Read a BAM file from S3 using the AWS CLI and pysam.

    This function attempts to deduce the AWS CLI profile from the S3 URI
    if not provided, using the ~/.aws/config file.
    It requires the AWS CLI to be installed and configured with SSO credentials.

    :param bam_file: S3 URI, e.g. "s3://…/file.bam"
    :param profile: AWS CLI profile; if None, attempt to deduce from ~/.aws/config
    :returns: an open pysam.AlignmentFile (caller must close it)
    """
    return _read_file_from_s3(bam_file, profile, SUPPORTED_BAM_EXTENSIONS, pysam.AlignmentFile, "r")


def read_vcf_from_s3(
    vcf_file: str,
    profile: str | None = None,
) -> pysam.VariantFile:
    """
    Read a VCF file from S3 using the AWS CLI and pysam.

    This function attempts to deduce the AWS CLI profile from the S3 URI
    if not provided, using the ~/.aws/config file.
    It requires the AWS CLI to be installed and configured with SSO credentials.

    :param vcf_file: S3 URI, e.g. "s3://…/file.vcf.gz" or "s3://…/file.vcf"
    :param profile: AWS CLI profile; if None, attempt to deduce from ~/.aws/config
    :returns: an open pysam.VariantFile (caller must close it)
    """
    return _read_file_from_s3(vcf_file, profile, SUPPORTED_VCF_EXTENSIONS, pysam.VariantFile, "r")
