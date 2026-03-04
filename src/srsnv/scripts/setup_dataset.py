#!/usr/bin/env python3
"""Configure the shared SRSNV training workspace for a given dataset.

Auto-detects sample basename and mean coverage from the dataset folder,
creates symlinks in the shared workspace inputs directory, and writes
a ``dataset_config.json`` consumed by downstream scripts.

Usage
-----
    uv run python src/srsnv/scripts/setup_dataset.py /data/Runs/perchik/ppmseq_data/nanoseq_cord_blood

    # Override mean coverage or CRAM path:
    uv run python src/srsnv/scripts/setup_dataset.py /data/Runs/perchik/ppmseq_data/nanoseq_cord_blood \
        --mean-coverage 74 --cram /path/to/override.cram
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORKSPACE_DEFAULT = Path("/data/Runs/perchik/ppmseq_data/srsnv_training_workspace")
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
TRAINING_REGIONS = (
    REPO_ROOT
    / "src"
    / "srsnv"
    / "tests"
    / "resources"
    / "wgs_calling_regions.without_encode_blacklist.hg38.interval_list.gz"
)


def detect_basename(dataset_dir: Path) -> str:
    """Derive sample basename from parquet or CRAM files in *dataset_dir*.

    Parquets are checked first because the CRAM may carry a different
    sample ID (e.g. different barcode) while parquets, stats, and H5
    files all share the authoritative basename.
    """
    parquets = list((dataset_dir / "raw_filtered_featuremap_parquet").glob("*.parquet"))
    if parquets:
        name = parquets[0].stem
        for suffix in (".raw.featuremap.filtered",):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
        return name

    crams = list(dataset_dir.glob("*.cram"))
    if crams:
        return crams[0].stem

    raise FileNotFoundError(f"Cannot detect basename: no parquet or CRAM files found in {dataset_dir}")


def detect_mean_coverage(dataset_dir: Path, basename: str) -> float:
    """Extract median training coverage from the application QC H5 file."""
    h5_path = dataset_dir / "application_qc_h5" / f"{basename}.single_read_snv.applicationQC.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Application QC H5 not found: {h5_path}")

    import pandas as pd  # noqa: PLC0415

    with pd.HDFStore(str(h5_path), "r") as store:
        info = store["run_info_table"]
    cov = info.loc["Median training coverage"].iloc[0]
    return float(cov)


def resolve_paths(dataset_dir: Path, basename: str, cram_override: str | None) -> dict[str, Path]:
    """Build a dict of canonical input names to their resolved source paths."""
    paths: dict[str, Path] = {
        "positive.parquet": dataset_dir
        / "random_sample_filtered_featuremap_parquet"
        / f"{basename}.random_sample.featuremap.filtered.parquet",
        "negative.parquet": dataset_dir
        / "raw_filtered_featuremap_parquet"
        / f"{basename}.raw.featuremap.filtered.parquet",
        "stats_featuremap.json": dataset_dir / "raw_featuremap_stats" / f"{basename}.raw.featuremap.stats.json",
        "stats_positive.json": dataset_dir
        / "random_sample_featuremap_stats"
        / f"{basename}.random_sample.featuremap.stats.json",
        "stats_negative.json": dataset_dir
        / "random_sample_featuremap_stats"
        / f"{basename}.random_sample.featuremap.stats.json",
        "training_regions.interval_list.gz": TRAINING_REGIONS,
    }

    if cram_override:
        cram = Path(cram_override).resolve()
    else:
        crams = list(dataset_dir.glob("*.cram"))
        if crams:
            cram = crams[0]
        else:
            cram = None

    if cram is not None:
        paths["source.cram"] = cram
        crai = cram.with_suffix(".cram.crai")
        if not crai.exists():
            crai = Path(str(cram) + ".crai")
        if crai.exists():
            paths["source.cram.crai"] = crai

    return paths


def setup_workspace(
    dataset_dir: Path,
    workspace: Path,
    mean_coverage: float,
    basename: str,
    cram_override: str | None,
) -> Path:
    """Create symlinks and write dataset_config.json. Returns config path."""
    inputs_dir = workspace / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    source_paths = resolve_paths(dataset_dir, basename, cram_override)

    missing = [name for name, p in source_paths.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source files: {missing}")

    for link_name, target in source_paths.items():
        link = inputs_dir / link_name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target.resolve())
        print(f"  {link_name} -> {target}")

    config = {
        "dataset_dir": str(dataset_dir.resolve()),
        "basename": basename,
        "mean_coverage": mean_coverage,
        "cram_path": str(source_paths.get("source.cram", "")),
        "workspace": str(workspace.resolve()),
    }
    config_path = inputs_dir / "dataset_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    print(f"  dataset_config.json written ({config_path})")
    return config_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Configure SRSNV training workspace for a dataset")
    ap.add_argument("dataset_dir", type=Path, help="Path to dataset folder (e.g. .../ppmseq_data/nanoseq_cord_blood)")
    ap.add_argument(
        "--workspace",
        type=Path,
        default=WORKSPACE_DEFAULT,
        help=f"Shared workspace directory (default: {WORKSPACE_DEFAULT})",
    )
    ap.add_argument("--mean-coverage", type=float, default=None, help="Override auto-detected mean coverage")
    ap.add_argument("--cram", type=str, default=None, help="Override CRAM path (if not in dataset folder)")
    ap.add_argument("--basename", type=str, default=None, help="Override auto-detected sample basename")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset_dir = args.dataset_dir.resolve()

    if not dataset_dir.is_dir():
        print(f"ERROR: dataset directory does not exist: {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    basename = args.basename or detect_basename(dataset_dir)
    print(f"Dataset:  {dataset_dir}")
    print(f"Basename: {basename}")

    mean_coverage = args.mean_coverage
    if mean_coverage is None:
        mean_coverage = detect_mean_coverage(dataset_dir, basename)
    print(f"Coverage: {mean_coverage}")

    print(f"\nSetting up workspace: {args.workspace}")
    config_path = setup_workspace(dataset_dir, args.workspace, mean_coverage, basename, args.cram)
    print(f"\nDone. Config: {config_path}")


if __name__ == "__main__":
    main()
