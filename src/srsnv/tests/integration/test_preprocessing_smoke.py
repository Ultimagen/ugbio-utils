"""End-to-end smoke tests for the DNN preprocessing pipeline.

Uses small synthetic BAM + parquet fixtures (~50 reads per class).
Tests the full flow: cram_to_tensor_cache -> combine_and_split -> DataModule.
"""

from __future__ import annotations

import array as _array
import pickle
from pathlib import Path

import polars as pl
import pysam
import pytest
import torch

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_test_bam(bam_path: str, reads: list[dict]) -> str:
    """Create a sorted, indexed BAM with the given reads."""
    header = pysam.AlignmentHeader.from_dict(
        {
            "HD": {"VN": "1.6", "SO": "coordinate"},
            "SQ": [
                {"SN": "chr1", "LN": 10000},
                {"SN": "chr2", "LN": 10000},
                {"SN": "chr21", "LN": 10000},
                {"SN": "chr22", "LN": 10000},
            ],
        }
    )
    unsorted_path = bam_path + ".unsorted.bam"
    with pysam.AlignmentFile(unsorted_path, "wb", header=header) as out:
        for r in reads:
            seg = pysam.AlignedSegment(header)
            seg.query_name = r["name"]
            seg.reference_id = header.get_tid(r["chrom"])
            seg.reference_start = r["pos"] - 1
            seg.query_sequence = r["seq"]
            seg.query_qualities = pysam.qualitystring_to_array("I" * len(r["seq"]))
            seg.cigar = [(0, len(r["seq"]))]
            seg.mapping_quality = r.get("mapq", 60)
            seg.flag = 0
            seq_len_r = len(r["seq"])
            seg.set_tag("MD", str(seq_len_r))
            seg.set_tag("tp", _array.array("i", list(range(seq_len_r))))
            seg.set_tag("t0", "A" * seq_len_r)
            seg.set_tag("rq", 0.9, "f")
            seg.set_tag("tm", "AQ")
            seg.set_tag("st", "PLUS")
            seg.set_tag("et", "MINUS")
            out.write(seg)
    pysam.sort("-o", bam_path, unsorted_path)
    pysam.index(bam_path)
    Path(unsorted_path).unlink(missing_ok=True)
    return bam_path


def _make_parquet(path: str, rows: list[dict]) -> str:
    frame = pl.DataFrame(rows)
    frame.write_parquet(path)
    return path


def _make_interval_list(path: str) -> str:
    lines = [
        "@HD\tVN:1.6\tSO:coordinate\n",
        "@SQ\tSN:chr1\tLN:10000\n",
        "@SQ\tSN:chr2\tLN:10000\n",
        "@SQ\tSN:chr21\tLN:10000\n",
        "@SQ\tSN:chr22\tLN:10000\n",
        "chr1\t1\t10000\t+\tregion1\n",
        "chr2\t1\t10000\t+\tregion2\n",
        "chr21\t1\t10000\t+\tregion3\n",
        "chr22\t1\t10000\t+\tregion4\n",
    ]
    with open(path, "w") as f:
        f.writelines(lines)
    return path


@pytest.fixture()
def synthetic_data(tmp_path):
    """Create a small synthetic BAM + positive/negative parquets.

    ~25 positive reads on chr1/chr2, ~25 negative reads on chr1/chr2,
    plus a few on chr21 (holdout).
    """
    reads = []
    pos_rows = []
    neg_rows = []
    seq_len = 20
    seq = "A" * seq_len

    for i in range(25):
        chrom = "chr1" if i < 15 else "chr2"
        pos = 100 + i * 10
        name = f"pos_read_{i}"
        reads.append({"name": name, "chrom": chrom, "pos": pos, "seq": seq})
        pos_rows.append(
            {
                "CHROM": chrom,
                "POS": pos,
                "REF": "A",
                "ALT": "T",
                "X_ALT": "T",
                "RN": name,
                "INDEX": i,
                "REV": 0,
                "MAPQ": 60,
                "rq": 0.9,
                "tm": "AQ",
                "st": "PLUS",
                "et": "MINUS",
                "EDIST": i % 3,
            }
        )

    for i in range(25):
        chrom = "chr1" if i < 15 else "chr2"
        pos = 500 + i * 10
        name = f"neg_read_{i}"
        reads.append({"name": name, "chrom": chrom, "pos": pos, "seq": seq})
        neg_rows.append(
            {
                "CHROM": chrom,
                "POS": pos,
                "REF": "A",
                "ALT": "T",
                "X_ALT": "T",
                "RN": name,
                "INDEX": i,
                "REV": 0,
                "MAPQ": 60,
                "rq": 0.9,
                "tm": "AQ",
                "st": "PLUS",
                "et": "MINUS",
                "EDIST": 0,
            }
        )

    for i in range(5):
        pos = 100 + i * 10
        name = f"holdout_pos_{i}"
        reads.append({"name": name, "chrom": "chr21", "pos": pos, "seq": seq})
        pos_rows.append(
            {
                "CHROM": "chr21",
                "POS": pos,
                "REF": "A",
                "ALT": "T",
                "X_ALT": "T",
                "RN": name,
                "INDEX": 100 + i,
                "REV": 0,
                "MAPQ": 60,
                "rq": 0.9,
                "tm": "AQ",
                "st": "PLUS",
                "et": "MINUS",
                "EDIST": i % 3,
            }
        )

    for i in range(5):
        pos = 500 + i * 10
        name = f"holdout_neg_{i}"
        reads.append({"name": name, "chrom": "chr21", "pos": pos, "seq": seq})
        neg_rows.append(
            {
                "CHROM": "chr21",
                "POS": pos,
                "REF": "A",
                "ALT": "T",
                "X_ALT": "T",
                "RN": name,
                "INDEX": 100 + i,
                "REV": 0,
                "MAPQ": 60,
                "rq": 0.9,
                "tm": "AQ",
                "st": "PLUS",
                "et": "MINUS",
                "EDIST": 0,
            }
        )

    bam_path = _make_test_bam(str(tmp_path / "test.bam"), reads)
    pos_parquet = _make_parquet(str(tmp_path / "positive.parquet"), pos_rows)
    neg_parquet = _make_parquet(str(tmp_path / "negative.parquet"), neg_rows)
    interval_list = _make_interval_list(str(tmp_path / "regions.interval_list"))

    return {
        "bam_path": bam_path,
        "pos_parquet": pos_parquet,
        "neg_parquet": neg_parquet,
        "interval_list": interval_list,
        "tmp_path": tmp_path,
        "n_positive": len(pos_rows),
        "n_negative": len(neg_rows),
    }


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestCramToTensorCacheSmoke:
    def test_basic(self, synthetic_data):
        from ugbio_srsnv.deep_srsnv.cram_to_tensors import cram_to_tensor_cache
        from ugbio_srsnv.deep_srsnv.data_prep import load_vocab_config

        enc = load_vocab_config()
        out_dir = str(synthetic_data["tmp_path"] / "pos_cache")

        index = cram_to_tensor_cache(
            cram_path=synthetic_data["bam_path"],
            parquet_path=synthetic_data["pos_parquet"],
            encoders=enc,
            output_dir=out_dir,
            label=True,
            tensor_length=50,
            shard_size=100,
        )

        out_path = Path(out_dir)
        assert (out_path / "index.json").exists()
        shard_files = list(out_path.glob("shard_*.pkl"))
        assert len(shard_files) >= 1

        with shard_files[0].open("rb") as f:
            chunk = pickle.load(f)

        assert "read_base_idx" in chunk
        assert "ref_base_idx" in chunk
        assert "mask" in chunk
        assert "label" in chunk
        assert chunk["read_base_idx"].shape[1] == 50
        assert chunk["label"].dtype == torch.uint8

        assert index["total_output_rows"] > 0
        assert index["total_output_rows"] <= synthetic_data["n_positive"]

        assert "profile" in index
        profile = index["profile"]
        assert profile["wall_seconds"] > 0
        assert profile["total_output_rows"] > 0


class TestCombineAndSplitSmokeKfold:
    def test_kfold(self, synthetic_data):
        from ugbio_srsnv.deep_srsnv.combine_splits import combine_and_split
        from ugbio_srsnv.deep_srsnv.cram_to_tensors import cram_to_tensor_cache
        from ugbio_srsnv.deep_srsnv.data_prep import load_vocab_config

        enc = load_vocab_config()
        tmp = synthetic_data["tmp_path"]

        pos_dir = str(tmp / "pos_cache_kfold")
        neg_dir = str(tmp / "neg_cache_kfold")

        cram_to_tensor_cache(
            cram_path=synthetic_data["bam_path"],
            parquet_path=synthetic_data["pos_parquet"],
            encoders=enc,
            output_dir=pos_dir,
            label=True,
            tensor_length=50,
            shard_size=100,
        )
        cram_to_tensor_cache(
            cram_path=synthetic_data["bam_path"],
            parquet_path=synthetic_data["neg_parquet"],
            encoders=enc,
            output_dir=neg_dir,
            label=False,
            tensor_length=50,
            shard_size=100,
        )

        folds_dir = str(tmp / "folds_kfold")
        index = combine_and_split(
            positive_cache_dir=pos_dir,
            negative_cache_dir=neg_dir,
            training_regions=synthetic_data["interval_list"],
            k_folds=2,
            holdout_chromosomes=["chr21"],
            random_seed=42,
            output_dir=folds_dir,
        )

        folds_path = Path(folds_dir)
        assert (folds_path / "fold_0").exists()
        assert (folds_path / "fold_1").exists()
        assert (folds_path / "fold_0" / "train.pt").exists()
        assert (folds_path / "fold_0" / "val.pt").exists()
        assert (folds_path / "fold_0" / "test.pt").exists()
        assert (folds_path / "split_manifest.json").exists()

        with (folds_path / "fold_0" / "test.pt").open("rb") as f:
            test_cache = pickle.load(f)
        chrom_col = test_cache["chrom"]
        test_chroms = set(chrom_col) if isinstance(chrom_col, list) else set(chrom_col.tolist())
        assert test_chroms == {"chr21"}, f"Test set should only contain chr21, got {test_chroms}"

        fold_0_info = index["fold_summary"][0]
        total_non_holdout = fold_0_info["train_rows"] + fold_0_info["val_rows"]
        assert total_non_holdout > 0

        assert "profile" in index


class TestCombineAndSplitSmokeSingleModel:
    def test_single_model(self, synthetic_data):
        from ugbio_srsnv.deep_srsnv.combine_splits import combine_and_split
        from ugbio_srsnv.deep_srsnv.cram_to_tensors import cram_to_tensor_cache
        from ugbio_srsnv.deep_srsnv.data_prep import load_vocab_config

        enc = load_vocab_config()
        tmp = synthetic_data["tmp_path"]

        pos_dir = str(tmp / "pos_cache_sm")
        neg_dir = str(tmp / "neg_cache_sm")

        cram_to_tensor_cache(
            cram_path=synthetic_data["bam_path"],
            parquet_path=synthetic_data["pos_parquet"],
            encoders=enc,
            output_dir=pos_dir,
            label=True,
            tensor_length=50,
            shard_size=100,
        )
        cram_to_tensor_cache(
            cram_path=synthetic_data["bam_path"],
            parquet_path=synthetic_data["neg_parquet"],
            encoders=enc,
            output_dir=neg_dir,
            label=False,
            tensor_length=50,
            shard_size=100,
        )

        folds_dir = str(tmp / "folds_sm")
        combine_and_split(
            positive_cache_dir=pos_dir,
            negative_cache_dir=neg_dir,
            training_regions=synthetic_data["interval_list"],
            holdout_chromosomes=["chr21"],
            val_chromosomes=["chr2"],
            single_model_split=True,
            random_seed=42,
            output_dir=folds_dir,
        )

        folds_path = Path(folds_dir)
        assert (folds_path / "fold_0").exists()
        assert not (folds_path / "fold_1").exists()

        with (folds_path / "fold_0" / "train.pt").open("rb") as f:
            train_cache = pickle.load(f)
        with (folds_path / "fold_0" / "val.pt").open("rb") as f:
            val_cache = pickle.load(f)
        with (folds_path / "fold_0" / "test.pt").open("rb") as f:
            test_cache = pickle.load(f)

        def _chrom_set(cache):
            c = cache["chrom"]
            return set(c) if isinstance(c, list) else set(c.tolist())

        train_chroms = _chrom_set(train_cache)
        val_chroms = _chrom_set(val_cache)
        test_chroms = _chrom_set(test_cache)

        assert "chr1" in train_chroms
        assert "chr2" in val_chroms
        assert "chr21" in test_chroms


class TestEndToEndDataModuleSmoke:
    def test_datamodule_from_fold_dir(self, synthetic_data):
        from ugbio_srsnv.deep_srsnv.combine_splits import combine_and_split
        from ugbio_srsnv.deep_srsnv.cram_to_tensors import cram_to_tensor_cache
        from ugbio_srsnv.deep_srsnv.data_module import SRSNVDataModule
        from ugbio_srsnv.deep_srsnv.data_prep import load_vocab_config

        enc = load_vocab_config()
        tmp = synthetic_data["tmp_path"]

        pos_dir = str(tmp / "pos_cache_e2e")
        neg_dir = str(tmp / "neg_cache_e2e")

        cram_to_tensor_cache(
            cram_path=synthetic_data["bam_path"],
            parquet_path=synthetic_data["pos_parquet"],
            encoders=enc,
            output_dir=pos_dir,
            label=True,
            tensor_length=50,
            shard_size=100,
        )
        cram_to_tensor_cache(
            cram_path=synthetic_data["bam_path"],
            parquet_path=synthetic_data["neg_parquet"],
            encoders=enc,
            output_dir=neg_dir,
            label=False,
            tensor_length=50,
            shard_size=100,
        )

        folds_dir = str(tmp / "folds_e2e")
        combine_and_split(
            positive_cache_dir=pos_dir,
            negative_cache_dir=neg_dir,
            training_regions=synthetic_data["interval_list"],
            k_folds=2,
            holdout_chromosomes=["chr21"],
            random_seed=42,
            output_dir=folds_dir,
        )

        dm = SRSNVDataModule.from_fold_dir(
            fold_dir=str(Path(folds_dir) / "fold_0"),
            train_batch_size=8,
        )
        dm.setup("fit")

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        assert isinstance(batch, dict)
        assert "read_base_idx" in batch
        assert "ref_base_idx" in batch
        assert "x_num" in batch
        assert "mask" in batch
        assert "label" in batch

        assert batch["read_base_idx"].dtype == torch.long
        assert batch["mask"].dtype == torch.float32
        assert batch["label"].dtype == torch.float32
        assert batch["x_num"].shape[1] == 9  # 5 positional + 4 constant
        assert batch["x_num"].shape[2] == 50  # tensor_length
