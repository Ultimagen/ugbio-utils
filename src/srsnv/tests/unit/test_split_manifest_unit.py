"""Unit tests for split_manifest module.

Covers:
- chromosome_kfold mode: partition_chromosomes_greedy, build_split_manifest
- single_model_read_hash mode: build, assign, _rn_hash_fraction
- single_model_chrom_val mode: build, assign
- parse_interval_list (manual fallback)
- validate_manifest_against_regions
- save/load manifest
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from unittest.mock import patch

import pytest
from ugbio_srsnv.split_manifest import (
    SPLIT_MODE_CHROM_KFOLD,
    SPLIT_MODE_SINGLE_MODEL_CHROM_VAL,
    SPLIT_MODE_SINGLE_MODEL_READ_HASH,
    _parse_interval_list_manual,
    _pick_smallest_chroms,
    _rn_hash_fraction,
    _validate_required_fields,
    assign_single_model_chrom_val_role,
    assign_single_model_read_hash_role,
    build_single_model_chrom_val_manifest,
    build_single_model_read_hash_manifest,
    build_split_manifest,
    load_split_manifest,
    parse_interval_list,
    partition_chromosomes_greedy,
    save_split_manifest,
    validate_manifest_against_regions,
)

# ──────────────────────── helpers ──────────────────────────────────


def _write_interval_list(path: str, chrom_sizes: dict[str, int], chroms_in_data: list[str], *, gzipped: bool = False):
    """Write a minimal interval-list file (with @SQ headers and data lines)."""
    lines = []
    for name, length in chrom_sizes.items():
        lines.append(f"@SQ\tSN:{name}\tLN:{length}\n")
    for chrom in chroms_in_data:
        lines.append(f"{chrom}\t1\t{chrom_sizes[chrom]}\t+\tinterval\n")

    if gzipped:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.writelines(lines)
    else:
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(lines)


@pytest.fixture
def interval_list_file(tmp_path):
    """Create a simple interval list with chr1-chr5."""
    chrom_sizes = {"chr1": 1000, "chr2": 800, "chr3": 600, "chr4": 400, "chr5": 200}
    chroms_in_data = list(chrom_sizes.keys())
    path = str(tmp_path / "regions.interval_list")
    _write_interval_list(path, chrom_sizes, chroms_in_data)
    return path, chrom_sizes, chroms_in_data


@pytest.fixture
def interval_list_file_gzipped(tmp_path):
    """Create a gzipped interval list."""
    chrom_sizes = {"chr1": 1000, "chr2": 800, "chr3": 600, "chr4": 400, "chr5": 200}
    chroms_in_data = list(chrom_sizes.keys())
    path = str(tmp_path / "regions.interval_list.gz")
    _write_interval_list(path, chrom_sizes, chroms_in_data, gzipped=True)
    return path, chrom_sizes, chroms_in_data


# ──────────────────────── parse_interval_list ──────────────────────


class TestParseIntervalList:
    """Test interval list parsing (manual fallback, since we don't have tabix index)."""

    def test_manual_plain_text(self, interval_list_file):
        path, expected_sizes, expected_chroms = interval_list_file
        chrom_sizes, chroms_in_data = parse_interval_list(path)
        assert chrom_sizes == expected_sizes
        assert chroms_in_data == expected_chroms

    def test_manual_gzipped(self, interval_list_file_gzipped):
        path, expected_sizes, expected_chroms = interval_list_file_gzipped
        chrom_sizes, chroms_in_data = _parse_interval_list_manual(path)
        assert chrom_sizes == expected_sizes
        assert chroms_in_data == expected_chroms

    def test_missing_sq_header_raises(self, tmp_path):
        """If data contigs have no matching @SQ header, raise ValueError."""
        path = str(tmp_path / "bad.interval_list")
        with open(path, "w") as fh:
            fh.write("@SQ\tSN:chr1\tLN:1000\n")
            fh.write("chr2\t1\t500\t+\tinterval\n")  # chr2 has no @SQ
        with pytest.raises(ValueError, match="Missing @SQ header for contigs"):
            _parse_interval_list_manual(path)

    def test_tabix_path_used_when_tbi_exists(self, tmp_path, interval_list_file):
        """If a .tbi file exists, parse_interval_list should try tabix path."""
        path, _, _ = interval_list_file
        tbi_path = path + ".tbi"
        Path(tbi_path).touch()
        # Since we don't have a real tabix file, it will fail;
        # just verify it attempts the tabix path
        with patch("ugbio_srsnv.split_manifest._parse_interval_list_tabix") as mock_tabix:
            mock_tabix.return_value = ({}, [])
            parse_interval_list(path)
            mock_tabix.assert_called_once_with(path)


# ──────────────────────── partition_chromosomes_greedy ──────────────


class TestPartitionChromosomesGreedy:
    """Test greedy chromosome partitioning."""

    def test_basic_partition(self):
        chrom_sizes = {"chr1": 100, "chr2": 80, "chr3": 60, "chr4": 40, "chr5": 20}
        chromosomes = list(chrom_sizes.keys())
        result = partition_chromosomes_greedy(chrom_sizes, chromosomes, k_folds=2)
        # Every chromosome should be assigned a fold (0 or 1)
        assert set(result.keys()) == set(chromosomes)
        assert all(v in (0, 1) for v in result.values())

    def test_single_fold(self):
        chrom_sizes = {"chr1": 100, "chr2": 50}
        result = partition_chromosomes_greedy(chrom_sizes, list(chrom_sizes.keys()), k_folds=1)
        # Everything in fold 0
        assert all(v == 0 for v in result.values())

    def test_balanced_partition(self):
        """With equal-size chromosomes, folds should be balanced."""
        chrom_sizes = {f"chr{i}": 100 for i in range(1, 5)}
        result = partition_chromosomes_greedy(chrom_sizes, list(chrom_sizes.keys()), k_folds=2)
        fold_counts = [0, 0]
        for v in result.values():
            fold_counts[v] += 1
        assert fold_counts == [2, 2]


# ──────────────────────── _pick_smallest_chroms ──────────────────────


class TestPickSmallestChroms:
    def test_picks_smallest(self):
        chrom_sizes = {"chr1": 1000, "chr2": 200, "chr3": 500, "chr4": 100}
        result = _pick_smallest_chroms(chrom_sizes, list(chrom_sizes.keys()), n_chroms_leave_out=2)
        assert set(result) == {"chr2", "chr4"}

    def test_single_chrom(self):
        chrom_sizes = {"chr1": 1000, "chr2": 200, "chr3": 500}
        result = _pick_smallest_chroms(chrom_sizes, list(chrom_sizes.keys()), n_chroms_leave_out=1)
        assert result == ["chr2"]


# ──────────────────────── build_split_manifest (chromosome_kfold) ──────


class TestBuildSplitManifest:
    def test_basic_build(self, interval_list_file):
        path, chrom_sizes, chrom_list = interval_list_file
        manifest = build_split_manifest(
            training_regions=path,
            k_folds=3,
            random_seed=42,
        )
        assert manifest["split_mode"] == SPLIT_MODE_CHROM_KFOLD
        assert manifest["split_version"] == 1
        assert manifest["k_folds"] == 3
        assert manifest["random_seed"] == 42
        # Test chroms should be the smallest
        assert len(manifest["test_chromosomes"]) >= 1
        # All chromosomes should be accounted for
        all_chroms = set(manifest["train_chromosomes"]) | set(manifest["test_chromosomes"])
        assert all_chroms == set(chrom_list)

    def test_explicit_holdout(self, interval_list_file):
        path, _, _ = interval_list_file
        manifest = build_split_manifest(
            training_regions=path,
            k_folds=2,
            random_seed=7,
            holdout_chromosomes=["chr1", "chr2"],
        )
        assert set(manifest["test_chromosomes"]) == {"chr1", "chr2"}
        assert "chr1" not in manifest["train_chromosomes"]
        assert "chr2" not in manifest["train_chromosomes"]

    def test_invalid_holdout_raises(self, interval_list_file):
        path, _, _ = interval_list_file
        with pytest.raises(ValueError, match="not found in interval list"):
            build_split_manifest(
                training_regions=path,
                k_folds=2,
                random_seed=7,
                holdout_chromosomes=["chrX"],
            )

    def test_val_chromosomes_per_fold(self, interval_list_file):
        path, _, _ = interval_list_file
        manifest = build_split_manifest(
            training_regions=path,
            k_folds=2,
            random_seed=42,
            holdout_chromosomes=["chr5"],
        )
        val_per_fold = manifest["val_chromosomes_per_fold"]
        # We have 2 folds
        assert "0" in val_per_fold and "1" in val_per_fold
        # Each chrom appears in exactly one fold
        all_val_chroms = set()
        for fold_chroms in val_per_fold.values():
            all_val_chroms.update(fold_chroms)
        # val chroms should be the train_chromosomes
        assert all_val_chroms == set(manifest["train_chromosomes"])


# ──────────────────────── single_model_read_hash ──────────────────────


class TestSingleModelReadHash:
    def test_build(self, interval_list_file):
        path, _, _ = interval_list_file
        manifest = build_single_model_read_hash_manifest(
            training_regions=path,
            random_seed=42,
            holdout_chromosomes=["chr5"],
            val_fraction=0.1,
        )
        assert manifest["split_mode"] == SPLIT_MODE_SINGLE_MODEL_READ_HASH
        assert manifest["val_fraction"] == 0.1
        assert manifest["hash_key"] == "RN"
        assert "chr5" in manifest["test_chromosomes"]
        assert "chr5" not in manifest["train_val_chromosomes"]

    def test_invalid_val_fraction(self, interval_list_file):
        path, _, _ = interval_list_file
        with pytest.raises(ValueError, match="val_fraction must be in"):
            build_single_model_read_hash_manifest(
                training_regions=path,
                random_seed=42,
                holdout_chromosomes=["chr5"],
                val_fraction=0.0,
            )

    def test_invalid_hash_key(self, interval_list_file):
        path, _, _ = interval_list_file
        with pytest.raises(ValueError, match="Only hash_key='RN'"):
            build_single_model_read_hash_manifest(
                training_regions=path,
                random_seed=42,
                holdout_chromosomes=["chr5"],
                hash_key="XX",
            )

    def test_no_holdout_raises(self, interval_list_file):
        path, _, _ = interval_list_file
        with pytest.raises(ValueError, match="holdout_chromosomes must be provided"):
            build_single_model_read_hash_manifest(
                training_regions=path,
                random_seed=42,
                holdout_chromosomes=[],
            )

    def test_assign_role_test_chrom(self):
        manifest = {
            "test_chromosomes": ["chr5"],
            "val_fraction": 0.1,
            "random_seed": 42,
        }
        assert assign_single_model_read_hash_role("chr5", "any_read", manifest) == "test"

    def test_assign_role_train_val(self):
        manifest = {
            "test_chromosomes": ["chr5"],
            "val_fraction": 0.1,
            "random_seed": 42,
        }
        # With val_fraction=0.1, most reads should be train
        roles = [assign_single_model_read_hash_role("chr1", f"read_{i}", manifest) for i in range(1000)]
        val_count = roles.count("val")
        train_count = roles.count("train")
        # Approximately 10% should be val (allow wide range for randomness)
        assert 50 < val_count < 200
        assert train_count > 700


class TestRnHashFraction:
    def test_deterministic(self):
        """Same inputs produce same output."""
        assert _rn_hash_fraction("readA", 42) == _rn_hash_fraction("readA", 42)

    def test_different_seed_different_result(self):
        """Different seeds produce different fractions."""
        assert _rn_hash_fraction("readA", 42) != _rn_hash_fraction("readA", 43)

    def test_range(self):
        """Output is in [0, 1)."""
        for i in range(100):
            frac = _rn_hash_fraction(f"read_{i}", 42)
            assert 0.0 <= frac < 1.0

    def test_empty_rn(self):
        """Empty read name does not crash."""
        frac = _rn_hash_fraction("", 42)
        assert 0.0 <= frac < 1.0

    def test_none_rn(self):
        """None read name handled as empty string."""
        frac = _rn_hash_fraction(None, 42)
        assert 0.0 <= frac < 1.0


# ──────────────────────── single_model_chrom_val ──────────────────────


class TestSingleModelChromVal:
    def test_build(self, interval_list_file):
        path, _, _ = interval_list_file
        manifest = build_single_model_chrom_val_manifest(
            training_regions=path,
            holdout_chromosomes=["chr5"],
            val_chromosomes=["chr4"],
        )
        assert manifest["split_mode"] == SPLIT_MODE_SINGLE_MODEL_CHROM_VAL
        assert "chr5" in manifest["test_chromosomes"]
        assert "chr4" in manifest["val_chromosomes"]
        assert "chr5" not in manifest["train_chromosomes"]
        assert "chr4" not in manifest["train_chromosomes"]

    def test_overlap_raises(self, interval_list_file):
        path, _, _ = interval_list_file
        with pytest.raises(ValueError, match="overlap"):
            build_single_model_chrom_val_manifest(
                training_regions=path,
                holdout_chromosomes=["chr5"],
                val_chromosomes=["chr5"],  # overlaps with holdout
            )

    def test_missing_holdout(self, interval_list_file):
        path, _, _ = interval_list_file
        with pytest.raises(ValueError, match="holdout_chromosomes must be provided"):
            build_single_model_chrom_val_manifest(
                training_regions=path,
                holdout_chromosomes=[],
                val_chromosomes=["chr4"],
            )

    def test_missing_val(self, interval_list_file):
        path, _, _ = interval_list_file
        with pytest.raises(ValueError, match="val_chromosomes must be provided"):
            build_single_model_chrom_val_manifest(
                training_regions=path,
                holdout_chromosomes=["chr5"],
                val_chromosomes=[],
            )

    def test_assign_role(self):
        manifest = {
            "test_chromosomes": ["chr5"],
            "val_chromosomes": ["chr4"],
            "train_chromosomes": ["chr1", "chr2", "chr3"],
        }
        assert assign_single_model_chrom_val_role("chr5", manifest) == "test"
        assert assign_single_model_chrom_val_role("chr4", manifest) == "val"
        assert assign_single_model_chrom_val_role("chr1", manifest) == "train"
        assert assign_single_model_chrom_val_role("chr3", manifest) == "train"


# ──────────────────────── validation ──────────────────────────────────


class TestValidation:
    def test_validate_required_fields_kfold(self):
        manifest = {
            "split_version": 1,
            "random_seed": 42,
            "k_folds": 3,
            "holdout_chromosomes": [],
            "chrom_to_fold": {},
            "train_chromosomes": [],
            "val_chromosomes_per_fold": {},
            "test_chromosomes": [],
        }
        # Should not raise
        _validate_required_fields(manifest, SPLIT_MODE_CHROM_KFOLD)

    def test_validate_required_fields_read_hash(self):
        manifest = {
            "split_version": 1,
            "random_seed": 42,
            "holdout_chromosomes": [],
            "test_chromosomes": [],
            "train_val_chromosomes": [],
            "val_fraction": 0.1,
            "hash_key": "RN",
        }
        # Should not raise
        _validate_required_fields(manifest, SPLIT_MODE_SINGLE_MODEL_READ_HASH)

    def test_validate_required_fields_chrom_val(self):
        manifest = {
            "split_version": 1,
            "holdout_chromosomes": [],
            "test_chromosomes": [],
            "val_chromosomes": [],
            "train_chromosomes": [],
        }
        # Should not raise
        _validate_required_fields(manifest, SPLIT_MODE_SINGLE_MODEL_CHROM_VAL)

    def test_validate_required_fields_missing(self):
        manifest = {"split_version": 1}
        with pytest.raises(ValueError, match="Missing required manifest field"):
            _validate_required_fields(manifest, SPLIT_MODE_CHROM_KFOLD)

    def test_validate_required_fields_read_hash_missing(self):
        manifest = {"split_version": 1, "random_seed": 42}
        with pytest.raises(ValueError, match="Missing required manifest field"):
            _validate_required_fields(manifest, SPLIT_MODE_SINGLE_MODEL_READ_HASH)

    def test_validate_required_fields_chrom_val_missing(self):
        manifest = {"split_version": 1}
        with pytest.raises(ValueError, match="Missing required manifest field"):
            _validate_required_fields(manifest, SPLIT_MODE_SINGLE_MODEL_CHROM_VAL)

    def test_validate_unknown_mode(self):
        with pytest.raises(ValueError, match="Unknown split_mode"):
            _validate_required_fields({}, "unknown_mode")

    def test_validate_manifest_roundtrip(self, interval_list_file):
        path, _, _ = interval_list_file
        manifest = build_split_manifest(
            training_regions=path,
            k_folds=2,
            random_seed=42,
            holdout_chromosomes=["chr5"],
        )
        # Should not raise
        validate_manifest_against_regions(manifest, path)

    def test_validate_manifest_bad_chromosome(self, interval_list_file):
        path, _, _ = interval_list_file
        manifest = build_split_manifest(
            training_regions=path,
            k_folds=2,
            random_seed=42,
            holdout_chromosomes=["chr5"],
        )
        manifest["test_chromosomes"] = ["chrX"]
        with pytest.raises(ValueError, match="absent from interval list"):
            validate_manifest_against_regions(manifest, path)

    def test_validate_read_hash_manifest(self, interval_list_file):
        """validate_manifest_against_regions works for read-hash mode."""
        path, _, _ = interval_list_file
        manifest = build_single_model_read_hash_manifest(
            training_regions=path,
            random_seed=42,
            holdout_chromosomes=["chr5"],
            val_fraction=0.1,
        )
        # Should not raise
        validate_manifest_against_regions(manifest, path)

    def test_validate_chrom_val_manifest(self, interval_list_file):
        """validate_manifest_against_regions works for chrom-val mode."""
        path, _, _ = interval_list_file
        manifest = build_single_model_chrom_val_manifest(
            training_regions=path,
            holdout_chromosomes=["chr5"],
            val_chromosomes=["chr4"],
        )
        # Should not raise
        validate_manifest_against_regions(manifest, path)

    def test_validate_kfold_invalid_fold_ids(self, interval_list_file):
        """Fold IDs out of range should fail validation."""
        path, _, _ = interval_list_file
        manifest = build_split_manifest(
            training_regions=path,
            k_folds=2,
            random_seed=42,
            holdout_chromosomes=["chr5"],
        )
        # Corrupt fold ids
        for chrom in manifest["chrom_to_fold"]:
            manifest["chrom_to_fold"][chrom] = 99  # out of range
        with pytest.raises(ValueError, match="fold ids out of range"):
            validate_manifest_against_regions(manifest, path)

    def test_validate_kfold_test_train_overlap(self, interval_list_file):
        """Test and train chromosomes overlapping should fail."""
        path, _, _ = interval_list_file
        manifest = build_split_manifest(
            training_regions=path,
            k_folds=2,
            random_seed=42,
            holdout_chromosomes=["chr5"],
        )
        # Add test chromosome to train list
        manifest["train_chromosomes"].append("chr5")
        with pytest.raises(ValueError, match="overlap"):
            validate_manifest_against_regions(manifest, path)

    def test_validate_read_hash_bad_val_fraction(self, interval_list_file):
        """Invalid val_fraction in manifest should fail validation."""
        path, _, _ = interval_list_file
        manifest = build_single_model_read_hash_manifest(
            training_regions=path,
            random_seed=42,
            holdout_chromosomes=["chr5"],
            val_fraction=0.1,
        )
        manifest["val_fraction"] = 1.5  # invalid
        with pytest.raises(ValueError, match="val_fraction must be in"):
            validate_manifest_against_regions(manifest, path)

    def test_validate_read_hash_bad_hash_key(self, interval_list_file):
        """Invalid hash_key in manifest should fail validation."""
        path, _, _ = interval_list_file
        manifest = build_single_model_read_hash_manifest(
            training_regions=path,
            random_seed=42,
            holdout_chromosomes=["chr5"],
            val_fraction=0.1,
        )
        manifest["hash_key"] = "XX"
        with pytest.raises(ValueError, match="hash_key must be 'RN'"):
            validate_manifest_against_regions(manifest, path)

    def test_validate_read_hash_overlap_test_train(self, interval_list_file):
        """Overlapping test and train_val chromosomes should fail."""
        path, _, _ = interval_list_file
        manifest = build_single_model_read_hash_manifest(
            training_regions=path,
            random_seed=42,
            holdout_chromosomes=["chr5"],
            val_fraction=0.1,
        )
        manifest["train_val_chromosomes"].append("chr5")
        with pytest.raises(ValueError, match="overlap"):
            validate_manifest_against_regions(manifest, path)

    def test_validate_chrom_val_val_train_overlap(self, interval_list_file):
        """Overlapping val and train chromosomes should fail."""
        path, _, _ = interval_list_file
        manifest = build_single_model_chrom_val_manifest(
            training_regions=path,
            holdout_chromosomes=["chr5"],
            val_chromosomes=["chr4"],
        )
        manifest["train_chromosomes"].append("chr4")
        with pytest.raises(ValueError, match="overlap"):
            validate_manifest_against_regions(manifest, path)


# ──────────────────────── save/load ──────────────────────────────────


class TestSaveLoad:
    def test_roundtrip(self, tmp_path, interval_list_file):
        path, _, _ = interval_list_file
        manifest = build_split_manifest(
            training_regions=path,
            k_folds=2,
            random_seed=42,
            holdout_chromosomes=["chr5"],
        )
        out_path = tmp_path / "manifest.json"
        save_split_manifest(manifest, out_path)
        loaded = load_split_manifest(out_path)
        assert loaded == manifest

    def test_save_creates_parent_dirs(self, tmp_path):
        out_path = tmp_path / "sub" / "dir" / "manifest.json"
        save_split_manifest({"test": 1}, out_path)
        assert out_path.exists()
        loaded = json.loads(out_path.read_text())
        assert loaded == {"test": 1}
