"""Discover sample directories for multi-sample sorter stats reports."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from ugbio_core.logger import logger
from ugbio_core.sorter_stats_report import _find_sorter_basename, _list_s3_dir
from ugbio_core.sorter_utils import get_base_coverage_from_sorter, read_sorter_statistics_csv

try:
    import boto3
except ImportError:
    boto3 = None

try:
    from ugbio_cloud_utils.cloud_sync import cloud_sync
except ImportError:
    cloud_sync = None

_MIN_NAMED_SEGMENTS = 3


@dataclass
class SampleData:
    label: str
    stats_json: dict
    csv_df: pd.DataFrame
    base_coverage: dict


def _derive_sample_label(basename: str) -> str:
    """Strip the leading run ID prefix from a basename."""
    parts = basename.split("-", 1)
    if len(parts) > 1:
        return parts[1]
    return basename


def _has_named_sample(dirname: str) -> bool:
    """Check if a directory name has a sample name (RUNID-NAME-BARCODE-SEQ pattern).

    Junk folders follow RUNID-BARCODE-BARCODE_SEQ (2 segments after run ID).
    Real samples follow RUNID-SAMPLENAME-BARCODE-BARCODE_SEQ (3+ segments).
    """
    parts = dirname.split("-", 1)
    if len(parts) < 2:  # noqa: PLR2004
        return False
    after_runid = parts[1]
    segments = after_runid.split("-")
    return len(segments) >= _MIN_NAMED_SEGMENTS


def _parse_library_info_xml(xml_path: Path) -> set[str]:
    """Parse LibraryInfo XML and return the set of expected sample folder suffixes.

    Each sample maps to a folder like: RUNID-SAMPLENAME-INDEX_LABEL-INDEX_SEQUENCE
    Returns the set of "SAMPLENAME-INDEX_LABEL-INDEX_SEQUENCE" strings.
    """
    try:
        tree = ET.parse(xml_path)  # noqa: S314
    except ET.ParseError:
        logger.warning(f"Failed to parse LibraryInfo XML: {xml_path}")
        return set()

    root = tree.getroot()
    suffixes = set()
    for sample_elem in root.iter("Sample"):
        sample_id = sample_elem.get("Id", "")
        name = sample_id.split("@")[0] if "@" in sample_id else sample_id
        index_label = sample_elem.get("Index_Label", "")
        index_seq = sample_elem.get("Index_Sequence", "")
        if name and index_label and index_seq:
            suffixes.add(f"{name}-{index_label}-{index_seq}")
    return suffixes


def _discover_samples_local(run_dir: Path) -> list[Path]:
    """Discover sample directories in a local run directory."""
    xml_files = list(run_dir.glob("*_LibraryInfo.xml"))
    allowed_suffixes: set[str] | None = None
    if xml_files:
        logger.info(f"Using sample sheet: {xml_files[0].name}")
        allowed_suffixes = _parse_library_info_xml(xml_files[0])
        if not allowed_suffixes:
            allowed_suffixes = None

    sample_dirs = []
    for d in sorted(run_dir.iterdir()):
        if not d.is_dir():
            continue
        dirname = d.name
        if allowed_suffixes is not None:
            parts = dirname.split("-", 1)
            suffix = parts[1] if len(parts) > 1 else ""
            if suffix not in allowed_suffixes:
                continue
        elif not _has_named_sample(dirname):
            continue

        files = [f.name for f in d.iterdir() if f.is_file()]
        json_files = [f for f in files if f.endswith(".json") and "applicationQC" not in f and "_unmatched" not in f]
        csv_files = {f.removesuffix(".csv") for f in files if f.endswith(".csv")}
        if any(f.removesuffix(".json") in csv_files for f in json_files):
            sample_dirs.append(d)
            if allowed_suffixes is not None and len(sample_dirs) == len(allowed_suffixes):
                break
    return sample_dirs


def _find_s3_library_info_xml(s3_uri: str, all_files: list[str]) -> set[str] | None:
    """Look for a LibraryInfo XML in the S3 file listing and parse it if found."""
    xml_files = [f for f in all_files if f.endswith("_LibraryInfo.xml")]
    if not xml_files:
        return None

    if cloud_sync is None:
        return None

    xml_name = xml_files[0]
    logger.info(f"Using sample sheet: {xml_name}")
    local_path = Path(cloud_sync(f"{s3_uri.rstrip('/')}/{xml_name}"))
    result = _parse_library_info_xml(local_path)
    return result if result else None


def _filter_subdirs(subdirs: list[tuple[str, str]], allowed_suffixes: set[str] | None) -> list[str]:
    """Filter subdirectory list by allowed suffixes or name pattern."""
    filtered = []
    for dirname, full_uri in subdirs:
        if allowed_suffixes is not None:
            dir_parts = dirname.split("-", 1)
            suffix = dir_parts[1] if len(dir_parts) > 1 else ""
            if suffix not in allowed_suffixes:
                continue
        elif not _has_named_sample(dirname):
            continue
        filtered.append(full_uri)
    return filtered


def _list_s3_subdirs(s3_uri: str) -> list[str]:
    """List and filter S3 subdirectories for a run."""
    if boto3 is None:
        raise RuntimeError("No S3 access from this environment")

    uri = s3_uri.rstrip("/")
    parts = uri.replace("s3://", "").split("/", 1)
    bucket = parts[0]
    prefix = (parts[1] + "/") if len(parts) > 1 else ""

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")

    all_files: list[str] = []
    subdirs: list[tuple[str, str]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            filename = key[len(prefix) :]
            if filename:
                all_files.append(filename)
        for cp in page.get("CommonPrefixes", []):
            subdir_key = cp["Prefix"]
            subdir_name = subdir_key[len(prefix) :].rstrip("/")
            if subdir_name:
                subdirs.append((subdir_name, f"s3://{bucket}/{subdir_key}"))

    allowed_suffixes = _find_s3_library_info_xml(s3_uri, all_files)

    return _filter_subdirs(subdirs, allowed_suffixes)


def _load_sample(sample_dir: str | Path) -> SampleData | None:
    """Load a single sample's data from a local or S3 directory."""
    sample_dir_str = str(sample_dir)
    if sample_dir_str.startswith("s3://"):
        if cloud_sync is None:
            raise RuntimeError("No S3 access from this environment")

        all_files = _list_s3_dir(sample_dir_str)
        try:
            basename = _find_sorter_basename(all_files)
        except ValueError:
            return None
        json_name = f"{basename}.json"
        csv_name = f"{basename}.csv"
        if json_name not in all_files or csv_name not in all_files:
            return None
        json_path = Path(cloud_sync(f"{sample_dir_str.rstrip('/')}/{json_name}"))
        csv_path = Path(cloud_sync(f"{sample_dir_str.rstrip('/')}/{csv_name}"))
    else:
        dir_path = Path(sample_dir_str)
        all_files = [f.name for f in dir_path.iterdir() if f.is_file()]
        try:
            basename = _find_sorter_basename(all_files)
        except ValueError:
            return None
        json_path = dir_path / f"{basename}.json"
        csv_path = dir_path / f"{basename}.csv"
        if not json_path.exists() or not csv_path.exists():
            return None

    with open(json_path, encoding="utf-8") as f:
        stats_json = json.load(f)
    csv_df = read_sorter_statistics_csv(str(csv_path), edit_metric_names=False, as_dataframe=True)
    base_coverage = get_base_coverage_from_sorter(str(json_path))
    label = _derive_sample_label(basename)

    return SampleData(label=label, stats_json=stats_json, csv_df=csv_df, base_coverage=base_coverage)


def resolve_and_load_samples(run_dir: str | None, input_dirs: list[str] | None) -> list[SampleData]:
    """Resolve sample directories and load all valid samples."""
    sample_dirs: list[str] = []

    if run_dir:
        if run_dir.startswith("s3://"):
            sample_dirs.extend(_list_s3_subdirs(run_dir))
        else:
            sample_dirs.extend(str(d) for d in _discover_samples_local(Path(run_dir)))

    if input_dirs:
        sample_dirs.extend(input_dirs)

    if not sample_dirs:
        raise ValueError("No sample directories found. Provide --run-dir or --input-dir.")

    samples = []
    for sd in sample_dirs:
        logger.info(f"Loading sample from {sd}")
        sample = _load_sample(sd)
        if sample is not None:
            samples.append(sample)
        else:
            logger.warning(f"Skipping {sd}: no valid JSON+CSV pair found")

    if not samples:
        raise ValueError("No valid samples found in any of the provided directories.")

    return samples
