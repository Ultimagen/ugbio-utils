"""Discover sample directories for multi-sample sequencing QC reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from defusedxml.ElementTree import ParseError, parse
from ugbio_core.logger import logger
from ugbio_core.sorter_utils import get_base_coverage_from_sorter, read_sorter_statistics_csv

from ugbio_seq_qc.file_resolution import fetch_s3_file, list_s3_subdirs, resolve_sample_files

_MIN_NAMED_SEGMENTS = 3


@dataclass
class SampleData:
    label: str
    stats_json: dict
    csv_df: pd.DataFrame
    base_coverage: dict


def derive_sample_label(basename: str) -> str:
    """Strip the leading run ID prefix from a basename."""
    parts = basename.split("-", 1)
    if len(parts) > 1:
        return parts[1]
    return basename


def has_named_sample(dirname: str) -> bool:
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


def parse_library_info_xml(xml_path: Path) -> set[str]:
    """Parse a LibraryInfo XML and return the set of expected sample folder suffixes.

    Each sample maps to a folder like: RUNID-SAMPLENAME-INDEX_LABEL-INDEX_SEQUENCE
    Returns the set of "SAMPLENAME-INDEX_LABEL-INDEX_SEQUENCE" strings. Parsing uses
    defusedxml so external entities are not resolved; a malformed file logs a warning
    and yields an empty set rather than crashing.
    """
    try:
        tree = parse(xml_path)
    except ParseError:
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


def _matches_allowed(dirname: str, allowed_suffixes: set[str] | None) -> bool:
    """Decide whether a directory name should be kept for sample discovery."""
    if allowed_suffixes is not None:
        parts = dirname.split("-", 1)
        suffix = parts[1] if len(parts) > 1 else ""
        return suffix in allowed_suffixes
    return has_named_sample(dirname)


def discover_samples_local(run_dir: Path) -> list[Path]:
    """Discover sample directories in a local run directory."""
    xml_files = list(run_dir.glob("*_LibraryInfo.xml"))
    allowed_suffixes: set[str] | None = None
    if xml_files:
        logger.info(f"Using sample sheet: {xml_files[0].name}")
        allowed_suffixes = parse_library_info_xml(xml_files[0]) or None

    sample_dirs = []
    for d in sorted(run_dir.iterdir()):
        if not d.is_dir() or not _matches_allowed(d.name, allowed_suffixes):
            continue

        files = [f.name for f in d.iterdir() if f.is_file()]
        json_files = [f for f in files if f.endswith(".json") and "applicationQC" not in f and "_unmatched" not in f]
        csv_files = {f.removesuffix(".csv") for f in files if f.endswith(".csv")}
        if any(f.removesuffix(".json") in csv_files for f in json_files):
            sample_dirs.append(d)
            if allowed_suffixes is not None and len(sample_dirs) == len(allowed_suffixes):
                break
    return sample_dirs


def find_and_parse_s3_library_info(s3_uri: str, all_files: list[str]) -> set[str] | None:
    """Find a LibraryInfo XML in an S3 file listing, fetch it, and parse it.

    Returns the set of allowed sample suffixes, or None if no XML is present or it
    yields no usable entries.
    """
    xml_files = [f for f in all_files if f.endswith("_LibraryInfo.xml")]
    if not xml_files:
        return None

    xml_name = xml_files[0]
    logger.info(f"Using sample sheet: {xml_name}")
    local_path = fetch_s3_file(f"{s3_uri.rstrip('/')}/{xml_name}")
    return parse_library_info_xml(local_path) or None


def list_s3_sample_dirs(s3_uri: str) -> list[str]:
    """List and filter S3 subdirectories of a run, returning sample directory URIs."""
    subdirs, all_files = list_s3_subdirs(s3_uri)
    allowed_suffixes = find_and_parse_s3_library_info(s3_uri, all_files)
    return [uri for name, uri in subdirs if _matches_allowed(name, allowed_suffixes)]


def load_sample(sample_dir: str | Path) -> SampleData | None:
    """Load a single sample's data from a local or S3 directory."""
    try:
        json_path, csv_path, basename = resolve_sample_files(sample_dir)
    except ValueError:
        return None

    with open(json_path, encoding="utf-8") as f:
        stats_json = json.load(f)
    csv_df = read_sorter_statistics_csv(str(csv_path), edit_metric_names=False, as_dataframe=True)
    base_coverage = get_base_coverage_from_sorter(str(json_path))
    label = derive_sample_label(basename)

    return SampleData(label=label, stats_json=stats_json, csv_df=csv_df, base_coverage=base_coverage)


def resolve_and_load_samples(run_dir: str | None, input_dirs: list[str] | None) -> list[SampleData]:
    """Resolve sample directories and load all valid samples."""
    sample_dirs: list[str] = []

    if run_dir:
        if run_dir.startswith("s3://"):
            sample_dirs.extend(list_s3_sample_dirs(run_dir))
        else:
            sample_dirs.extend(str(d) for d in discover_samples_local(Path(run_dir)))

    if input_dirs:
        sample_dirs.extend(input_dirs)

    if not sample_dirs:
        raise ValueError("No sample directories found. Provide --run-dir or --input-dir.")

    samples = []
    for sd in sample_dirs:
        logger.info(f"Loading sample from {sd}")
        sample = load_sample(sd)
        if sample is not None:
            samples.append(sample)
        else:
            logger.warning(f"Skipping {sd}: no valid JSON+CSV pair found")

    if not samples:
        raise ValueError("No valid samples found in any of the provided directories.")

    return samples
