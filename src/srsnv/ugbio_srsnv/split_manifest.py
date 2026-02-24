from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pysam

SPLIT_MODE_CHROM_KFOLD = "chromosome_kfold"
SPLIT_MODE_SINGLE_MODEL_READ_HASH = "single_model_read_hash"


def _parse_interval_list_tabix(path: str) -> tuple[dict[str, int], list[str]]:
    chrom_sizes = {}
    with pysam.TabixFile(path) as tbx:
        for line in tbx.header:
            if line.startswith("@SQ"):
                chrom_name = None
                for field in line.strip().split("\t")[1:]:
                    key, val = field.split(":", 1)
                    if key == "SN":
                        chrom_name = val
                    elif key == "LN" and chrom_name is not None:
                        chrom_sizes[chrom_name] = int(val)
        chroms_in_data = list(tbx.contigs)
    missing = [c for c in chroms_in_data if c not in chrom_sizes]
    if missing:
        raise ValueError(f"Missing @SQ header for contigs: {missing}")
    return chrom_sizes, chroms_in_data


def _parse_interval_list_manual(path: str) -> tuple[dict[str, int], list[str]]:
    chrom_sizes: dict[str, int] = {}
    chroms_in_data: list[str] = []

    is_gzipped = path.endswith(".gz")
    if is_gzipped:
        fh = gzip.open(path, "rt", encoding="utf-8")
    else:
        fh = open(path, encoding="utf-8")

    try:
        for line in fh:
            if line.startswith("@SQ"):
                chrom_name = None
                for field in line.strip().split("\t")[1:]:
                    key, val = field.split(":", 1)
                    if key == "SN":
                        chrom_name = val
                    elif key == "LN" and chrom_name is not None:
                        chrom_sizes[chrom_name] = int(val)
            elif not line.startswith("@"):
                chrom = line.split("\t", 1)[0]
                if chrom not in chroms_in_data:
                    chroms_in_data.append(chrom)
    finally:
        fh.close()

    missing = [c for c in chroms_in_data if c not in chrom_sizes]
    if missing:
        raise ValueError(f"Missing @SQ header for contigs: {missing}")
    return chrom_sizes, chroms_in_data


def parse_interval_list(path: str) -> tuple[dict[str, int], list[str]]:
    candidate_tbi = path + ".tbi"
    if os.path.exists(candidate_tbi):
        return _parse_interval_list_tabix(path)
    return _parse_interval_list_manual(path)


def partition_chromosomes_greedy(
    chrom_sizes: dict[str, int],
    chromosomes: list[str],
    k_folds: int,
) -> dict[str, int]:
    series_of_sizes = pd.Series({c: chrom_sizes[c] for c in chromosomes}).sort_values(ascending=False)
    partitions = [[] for _ in range(k_folds)]
    partition_sums = np.zeros(k_folds)
    for idx, size in series_of_sizes.items():
        min_fold = partition_sums.argmin()
        partitions[min_fold].append(idx)
        partition_sums[min_fold] += size

    indices_to_folds = [[i for i, part in enumerate(partitions) if idx in part][0] for idx in series_of_sizes.index]
    return pd.Series(indices_to_folds, index=series_of_sizes.index).to_dict()


def _pick_smallest_chroms(chrom_sizes: dict[str, int], chromosomes: list[str], n_chroms_leave_out: int) -> list[str]:
    return list(
        pd.Series({c: chrom_sizes[c] for c in chromosomes}).sort_values(ascending=True).index[:n_chroms_leave_out]
    )


def build_split_manifest(
    training_regions: str,
    k_folds: int,
    random_seed: int,
    holdout_chromosomes: list[str] | None = None,
    n_chroms_leave_out: int = 1,
) -> dict:
    chrom_sizes, chrom_list = parse_interval_list(training_regions)
    if holdout_chromosomes:
        missing = [c for c in holdout_chromosomes if c not in chrom_list]
        if missing:
            raise ValueError(f"Holdout chromosome(s) not found in interval list: {missing}")
        test_chromosomes = list(dict.fromkeys(holdout_chromosomes))
    else:
        test_chromosomes = _pick_smallest_chroms(chrom_sizes, chrom_list, n_chroms_leave_out=n_chroms_leave_out)

    train_val_chromosomes = [c for c in chrom_list if c not in test_chromosomes]
    chrom_to_fold = partition_chromosomes_greedy(chrom_sizes, train_val_chromosomes, k_folds=k_folds)
    val_chromosomes_per_fold = {str(k): sorted([c for c, f in chrom_to_fold.items() if f == k]) for k in range(k_folds)}

    return {
        "split_version": 1,
        "split_mode": SPLIT_MODE_CHROM_KFOLD,
        "training_regions": training_regions,
        "random_seed": int(random_seed),
        "k_folds": int(k_folds),
        "holdout_chromosomes": test_chromosomes,
        "chrom_to_fold": chrom_to_fold,
        "train_chromosomes": sorted(train_val_chromosomes),
        "val_chromosomes_per_fold": val_chromosomes_per_fold,
        "test_chromosomes": sorted(test_chromosomes),
    }


def build_single_model_read_hash_manifest(
    training_regions: str,
    random_seed: int,
    holdout_chromosomes: list[str],
    val_fraction: float = 0.1,
    hash_key: str = "RN",
) -> dict:
    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if hash_key != "RN":
        raise ValueError(f"Only hash_key='RN' is currently supported, got {hash_key}")

    _, chrom_list = parse_interval_list(training_regions)
    if not holdout_chromosomes:
        raise ValueError("holdout_chromosomes must be provided for single-model read-hash split")
    missing = [c for c in holdout_chromosomes if c not in chrom_list]
    if missing:
        raise ValueError(f"Holdout chromosome(s) not found in interval list: {missing}")
    test_chromosomes = list(dict.fromkeys(holdout_chromosomes))
    train_val_chromosomes = [c for c in chrom_list if c not in test_chromosomes]
    return {
        "split_version": 1,
        "split_mode": SPLIT_MODE_SINGLE_MODEL_READ_HASH,
        "training_regions": training_regions,
        "random_seed": int(random_seed),
        "holdout_chromosomes": test_chromosomes,
        "test_chromosomes": sorted(test_chromosomes),
        "train_val_chromosomes": sorted(train_val_chromosomes),
        "val_fraction": float(val_fraction),
        "hash_key": hash_key,
    }


def _rn_hash_fraction(rn: str, random_seed: int) -> float:
    payload = f"{int(random_seed)}::{rn or ''}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return value / float(2**64)


def assign_single_model_read_hash_role(chrom: str, rn: str, manifest: dict) -> str:
    test_chromosomes = set(manifest.get("test_chromosomes", []))
    if chrom in test_chromosomes:
        return "test"
    val_fraction = float(manifest["val_fraction"])
    random_seed = int(manifest["random_seed"])
    return "val" if _rn_hash_fraction(rn=rn, random_seed=random_seed) < val_fraction else "train"


def validate_manifest_against_regions(manifest: dict, training_regions: str) -> None:  # noqa: C901, PLR0912
    split_mode = manifest.get("split_mode", SPLIT_MODE_CHROM_KFOLD)
    if split_mode == SPLIT_MODE_CHROM_KFOLD:
        required = {
            "split_version",
            "random_seed",
            "k_folds",
            "holdout_chromosomes",
            "chrom_to_fold",
            "train_chromosomes",
            "val_chromosomes_per_fold",
            "test_chromosomes",
        }
        missing_fields = sorted(required.difference(manifest))
        if missing_fields:
            raise ValueError(f"Missing required manifest field(s): {missing_fields}")
    elif split_mode == SPLIT_MODE_SINGLE_MODEL_READ_HASH:
        required = {
            "split_version",
            "random_seed",
            "holdout_chromosomes",
            "test_chromosomes",
            "train_val_chromosomes",
            "val_fraction",
            "hash_key",
        }
        missing_fields = sorted(required.difference(manifest))
        if missing_fields:
            raise ValueError(f"Missing required manifest field(s): {missing_fields}")
    else:
        raise ValueError(f"Unknown split_mode '{split_mode}'")

    chrom_sizes, chrom_list = parse_interval_list(training_regions)
    del chrom_sizes
    chrom_set = set(chrom_list)

    test_chroms = set(manifest["test_chromosomes"])
    unknown = sorted(test_chroms.difference(chrom_set))
    if unknown:
        raise ValueError(f"Manifest contains chromosome(s) absent from interval list: {unknown}")

    if split_mode == SPLIT_MODE_CHROM_KFOLD:
        chrom_to_fold = {str(k): int(v) for k, v in manifest["chrom_to_fold"].items()}
        train_chroms = set(manifest["train_chromosomes"])
        val_chroms = set(chrom_to_fold.keys())
        unknown = sorted((train_chroms | val_chroms).difference(chrom_set))
        if unknown:
            raise ValueError(f"Manifest contains chromosome(s) absent from interval list: {unknown}")
        if test_chroms & val_chroms:
            raise ValueError("Test and validation chromosomes overlap in split manifest")
        if test_chroms & train_chroms:
            raise ValueError("Test and train chromosomes overlap in split manifest")
        k_folds = int(manifest["k_folds"])
        fold_values = set(chrom_to_fold.values())
        invalid = sorted([f for f in fold_values if f < 0 or f >= k_folds])
        if invalid:
            raise ValueError(f"Manifest fold ids out of range [0, {k_folds - 1}]: {invalid}")
    else:
        train_val_chroms = set(manifest["train_val_chromosomes"])
        unknown = sorted(train_val_chroms.difference(chrom_set))
        if unknown:
            raise ValueError(f"Manifest contains chromosome(s) absent from interval list: {unknown}")
        if test_chroms & train_val_chroms:
            raise ValueError("Test and train/val chromosomes overlap in split manifest")
        val_fraction = float(manifest["val_fraction"])
        if not (0.0 < val_fraction < 1.0):
            raise ValueError(f"Manifest val_fraction must be in (0,1), got {val_fraction}")
        if manifest.get("hash_key") != "RN":
            raise ValueError("Manifest hash_key must be 'RN'")


def save_split_manifest(manifest: dict, path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def load_split_manifest(path: str | Path) -> dict:
    in_path = Path(path)
    return json.loads(in_path.read_text())
