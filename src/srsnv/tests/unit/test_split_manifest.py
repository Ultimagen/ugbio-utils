import json
from pathlib import Path

from ugbio_srsnv.split_manifest import (
    assign_single_model_read_hash_role,
    build_single_model_read_hash_manifest,
    build_split_manifest,
    load_split_manifest,
    save_split_manifest,
    validate_manifest_against_regions,
)


def _resources_dir() -> Path:
    return Path(__file__).parent.parent / "resources"


def test_build_split_manifest_with_explicit_holdout() -> None:
    interval_list = _resources_dir() / "wgs_calling_regions.without_encode_blacklist.hg38.interval_list.gz"
    manifest = build_split_manifest(
        training_regions=str(interval_list),
        k_folds=3,
        random_seed=11,
        holdout_chromosomes=["chr21", "chr22"],
    )
    assert manifest["k_folds"] == 3
    assert set(manifest["test_chromosomes"]) == {"chr21", "chr22"}
    assert "chr21" not in manifest["chrom_to_fold"]
    assert "chr22" not in manifest["chrom_to_fold"]
    validate_manifest_against_regions(manifest, str(interval_list))


def test_build_split_manifest_default_smallest_holdout() -> None:
    interval_list = _resources_dir() / "wgs_calling_regions.without_encode_blacklist.hg38.interval_list.gz"
    manifest = build_split_manifest(
        training_regions=str(interval_list),
        k_folds=2,
        random_seed=7,
        holdout_chromosomes=None,
    )
    # Keep old behavior by default: exactly one smallest chromosome excluded.
    assert len(manifest["test_chromosomes"]) == 1
    validate_manifest_against_regions(manifest, str(interval_list))


def test_roundtrip_save_and_load_split_manifest(tmp_path: Path) -> None:
    interval_list = _resources_dir() / "wgs_calling_regions.without_encode_blacklist.hg38.interval_list.gz"
    manifest = build_split_manifest(
        training_regions=str(interval_list),
        k_folds=2,
        random_seed=13,
        holdout_chromosomes=["chr21", "chr22"],
    )
    out_path = tmp_path / "split_manifest.json"
    save_split_manifest(manifest, out_path)
    loaded = load_split_manifest(out_path)
    assert loaded == json.loads(out_path.read_text())
    validate_manifest_against_regions(loaded, str(interval_list))


def test_single_model_read_hash_manifest_and_roles() -> None:
    interval_list = _resources_dir() / "wgs_calling_regions.without_encode_blacklist.hg38.interval_list.gz"
    manifest = build_single_model_read_hash_manifest(
        training_regions=str(interval_list),
        random_seed=13,
        holdout_chromosomes=["chr21", "chr22"],
        val_fraction=0.1,
        hash_key="RN",
    )
    validate_manifest_against_regions(manifest, str(interval_list))
    assert manifest["split_mode"] == "single_model_read_hash"

    role_test = assign_single_model_read_hash_role("chr21", "read_1", manifest)
    role_non_test = assign_single_model_read_hash_role("chr1", "read_1", manifest)
    role_non_test_repeat = assign_single_model_read_hash_role("chr1", "read_1", manifest)
    assert role_test == "test"
    assert role_non_test in {"train", "val"}
    assert role_non_test == role_non_test_repeat
