"""Shared S3 / local file-resolution helpers for sequencing QC reports.

This module is the single home for all S3 listing and sample file-resolution
logic used by both the single-sample and multi-sample reports. Because
``ugbio_seq_qc`` declares ``ugbio_cloud_utils`` (which pulls in ``boto3``) and
``cloud_sync`` as hard dependencies, these imports are unconditional at module
top level -- no ``try/except ImportError`` guards are needed.
"""

from __future__ import annotations

from pathlib import Path

import boto3
from ugbio_cloud_utils.cloud_sync import cloud_sync


def list_s3_dir(s3_uri: str) -> list[str]:
    """List filenames directly under an S3 "directory" (non-recursive).

    Parameters
    ----------
    s3_uri : str
        S3 URI of the directory, e.g. ``s3://bucket/path/to/dir/``.

    Returns
    -------
    list[str]
        Filenames (relative to the prefix) of objects directly under the prefix.
    """
    bucket, prefix = _split_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    files = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for obj in page.get("Contents", []):
            filename = obj["Key"][len(prefix) :]
            if filename:
                files.append(filename)
    return files


def list_s3_subdirs(s3_uri: str) -> tuple[list[tuple[str, str]], list[str]]:
    """List immediate subdirectories and files under an S3 "directory".

    Parameters
    ----------
    s3_uri : str
        S3 URI of the run directory, e.g. ``s3://bucket/path/to/run/``.

    Returns
    -------
    tuple[list[tuple[str, str]], list[str]]
        A tuple ``(subdirs, files)`` where ``subdirs`` is a list of
        ``(subdir_name, subdir_uri)`` pairs and ``files`` is the list of
        filenames directly under the prefix.
    """
    bucket, prefix = _split_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    files: list[str] = []
    subdirs: list[tuple[str, str]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for obj in page.get("Contents", []):
            filename = obj["Key"][len(prefix) :]
            if filename:
                files.append(filename)
        for cp in page.get("CommonPrefixes", []):
            subdir_key = cp["Prefix"]
            subdir_name = subdir_key[len(prefix) :].rstrip("/")
            if subdir_name:
                subdirs.append((subdir_name, f"s3://{bucket}/{subdir_key}"))
    return subdirs, files


def find_sample_basename(files: list[str]) -> str:
    """Find the main sample basename from a directory's file listing.

    Strategy:
    1. Look for the main CRAM file (excludes ``_unmatched`` and ``.crai``).
    2. If no CRAM is found, find JSON files that have a matching CSV (excludes
       ``applicationQC`` and ``_unmatched``).

    Parameters
    ----------
    files : list[str]
        Filenames present in the sample directory.

    Returns
    -------
    str
        The detected sample basename (filename without extension).

    Raises
    ------
    ValueError
        If a unique basename cannot be determined.
    """
    cram_files = [f for f in files if f.endswith(".cram") and "_unmatched" not in f and not f.endswith(".crai")]
    if len(cram_files) == 1:
        return cram_files[0].removesuffix(".cram")

    json_files = [
        f
        for f in files
        if f.endswith(".json") and "applicationQC" not in f and "_unmatched" not in f and not f.startswith(".")
    ]
    csv_files = {f.removesuffix(".csv") for f in files if f.endswith(".csv")}
    matched = [f.removesuffix(".json") for f in json_files if f.removesuffix(".json") in csv_files]

    if len(matched) == 1:
        return matched[0]
    if len(matched) > 1:
        raise ValueError(f"Multiple JSON+CSV pairs found, cannot auto-detect basename: {matched}")
    raise ValueError(
        f"Could not determine sample basename. No CRAM found ({len(cram_files)} candidates) "
        f"and no matching JSON+CSV pair found in: {files}"
    )


def resolve_sample_files(sample_dir: str | Path) -> tuple[Path, Path, str]:
    """Resolve the JSON and CSV stats files for a single sample directory.

    Works for both local paths and ``s3://`` URIs. For S3 inputs the JSON and
    CSV are fetched locally via ``cloud_sync``.

    Parameters
    ----------
    sample_dir : str | Path
        Local directory or ``s3://`` URI containing one sample's stats files.

    Returns
    -------
    tuple[Path, Path, str]
        ``(json_path, csv_path, basename)`` -- local paths to the JSON and CSV
        files and the detected sample basename.

    Raises
    ------
    ValueError
        If the JSON/CSV pair cannot be located.
    """
    sample_dir_str = str(sample_dir)
    if sample_dir_str.startswith("s3://"):
        all_files = list_s3_dir(sample_dir_str)
        basename = find_sample_basename(all_files)
        json_name = f"{basename}.json"
        csv_name = f"{basename}.csv"
        if json_name not in all_files or csv_name not in all_files:
            raise ValueError(f"Could not find {json_name} and/or {csv_name} in {sample_dir_str}")
        base_uri = sample_dir_str.rstrip("/")
        json_path = Path(cloud_sync(f"{base_uri}/{json_name}"))
        csv_path = Path(cloud_sync(f"{base_uri}/{csv_name}"))
    else:
        dir_path = Path(sample_dir_str)
        all_files = [f.name for f in dir_path.iterdir() if f.is_file()]
        basename = find_sample_basename(all_files)
        json_path = dir_path / f"{basename}.json"
        csv_path = dir_path / f"{basename}.csv"
        if not json_path.exists() or not csv_path.exists():
            raise ValueError(f"Could not find {json_path} and/or {csv_path}")
    return json_path, csv_path, basename


def fetch_s3_file(s3_uri: str) -> Path:
    """Fetch a single S3 object locally via ``cloud_sync`` and return its path."""
    return Path(cloud_sync(s3_uri))


def _split_s3_uri(s3_uri: str) -> tuple[str, str]:
    """Split an ``s3://bucket/prefix`` URI into ``(bucket, normalized_prefix)``.

    The returned prefix is empty (bucket root) or ends with a single ``/``.
    """
    parts = s3_uri.rstrip("/").replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = (parts[1] + "/") if len(parts) > 1 and parts[1] else ""
    return bucket, prefix
