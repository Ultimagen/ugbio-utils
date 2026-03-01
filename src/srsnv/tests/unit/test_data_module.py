import shutil
from pathlib import Path

import numpy as np
import torch
from ugbio_srsnv.deep_srsnv.data_module import SRSNVDataModule, TensorMapDataset, compact_collate_fn
from ugbio_srsnv.deep_srsnv.data_prep import load_cache_from_shm, save_cache_to_shm


def _make_fake_cache(n: int = 40, length: int = 300) -> dict:
    split_ids = []
    for i in range(n):
        if i % 5 == 4:
            split_ids.append(-1)
        else:
            split_ids.append(i % 3)

    return {
        "read_base_idx": torch.zeros(n, length, dtype=torch.int16),
        "ref_base_idx": torch.zeros(n, length, dtype=torch.int16),
        "t0_idx": torch.zeros(n, length, dtype=torch.int16),
        "tm_idx": torch.randint(0, 9, (n,), dtype=torch.int8),
        "st_idx": torch.randint(0, 5, (n,), dtype=torch.int8),
        "et_idx": torch.randint(0, 5, (n,), dtype=torch.int8),
        "x_num_pos": torch.randn(n, 5, length).to(dtype=torch.float16),
        "x_num_const": torch.randn(n, 4).to(dtype=torch.float16),
        "mask": torch.ones(n, length, dtype=torch.uint8),
        "label": torch.tensor([i % 2 for i in range(n)], dtype=torch.uint8),
        "split_id": torch.tensor(split_ids, dtype=torch.int8),
        "chrom": np.array([f"chr{(i % 5) + 1}" for i in range(n)], dtype=object),
        "pos": np.array([1000 + i for i in range(n)], dtype=np.int32),
        "rn": np.array([f"read_{i}" for i in range(n)], dtype=object),
    }


def test_tensor_map_dataset_split_filtering() -> None:
    cache = _make_fake_cache(40)
    ds = TensorMapDataset(cache, split_id_keep={0, 1})
    assert len(ds) > 0
    total_matching = int(torch.isin(cache["split_id"], torch.tensor([0, 1])).sum())
    assert len(ds) == total_matching


def test_tensor_map_dataset_with_meta() -> None:
    cache = _make_fake_cache(20)
    ds = TensorMapDataset(cache, split_id_keep={0, 1, 2, -1}, include_meta=True)
    item = ds[0]
    assert len(item) == 3
    gi, chrom, rn = item
    assert isinstance(gi, int)
    assert isinstance(chrom, str)
    assert isinstance(rn, str)


def test_compact_collate_fn() -> None:
    cache = _make_fake_cache(20)
    ds = TensorMapDataset(cache, split_id_keep={0, 1, 2, -1})
    batch_raw = [ds[i] for i in range(min(4, len(ds)))]
    result = compact_collate_fn(batch_raw, cache, include_meta=False)
    bs = len(batch_raw)
    assert result["read_base_idx"].shape == (bs, 300)
    assert result["x_num"].shape == (bs, 9, 300)
    assert result["label"].dtype == torch.float32
    assert result["mask"].dtype == torch.float32
    assert "tm_idx" in result
    assert result["tm_idx"].shape == (bs,)
    assert result["st_idx"].shape == (bs,)
    assert result["et_idx"].shape == (bs,)


def test_compact_collate_fn_with_meta() -> None:
    cache = _make_fake_cache(20)
    ds = TensorMapDataset(cache, split_id_keep={0, 1, 2, -1}, include_meta=True)
    batch_raw = [ds[i] for i in range(min(4, len(ds)))]
    result = compact_collate_fn(batch_raw, cache, include_meta=True)
    assert "chrom" in result
    assert "pos" in result
    assert "rn" in result
    assert len(result["chrom"]) == len(batch_raw)


def test_data_module_setup_and_dataloaders() -> None:
    cache = _make_fake_cache(40)
    dm = SRSNVDataModule(
        full_cache=cache,
        train_split_ids={0, 1},
        val_split_ids={2},
        test_split_ids={-1},
        train_batch_size=8,
    )
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    assert len(train_loader) > 0
    assert len(val_loader) > 0

    batch = next(iter(train_loader))
    assert "read_base_idx" in batch
    assert "label" in batch
    assert batch["x_num"].shape[1] == 9
    assert "tm_idx" in batch

    dm.setup("predict")
    pred_loader = dm.predict_dataloader()
    assert len(pred_loader) > 0
    pred_batch = next(iter(pred_loader))
    assert "chrom" in pred_batch
    assert "rn" in pred_batch


def test_data_module_no_batch_size_cap() -> None:
    cache = _make_fake_cache(20)
    dm = SRSNVDataModule(
        full_cache=cache,
        train_split_ids={0},
        val_split_ids={1},
        train_batch_size=4096,
    )
    assert dm.train_batch_size == 4096


def test_save_load_cache_shm_roundtrip(tmp_path: Path) -> None:
    """Verify save_cache_to_shm -> load_cache_from_shm produces identical data."""
    cache = _make_fake_cache(20)
    shm_dir = tmp_path / "fake_shm"
    shm_dir.mkdir()

    try:
        shm_path = save_cache_to_shm(cache, shm_dir=str(shm_dir))
        assert (shm_path / ".ready").exists()
        assert (shm_path / "tensors.pt").exists()
        assert (shm_path / "meta.pkl").exists()

        loaded = load_cache_from_shm(shm_path)

        for key in [
            "read_base_idx",
            "ref_base_idx",
            "t0_idx",
            "tm_idx",
            "st_idx",
            "et_idx",
            "x_num_pos",
            "x_num_const",
            "mask",
            "label",
            "split_id",
        ]:
            assert key in loaded, f"Missing key: {key}"
            assert torch.equal(cache[key], loaded[key]), f"Mismatch in tensor {key}"

        np.testing.assert_array_equal(cache["pos"], loaded["pos"])
        assert cache["chrom"].tolist() == list(loaded["chrom"])
        assert cache["rn"].tolist() == list(loaded["rn"])
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)


def test_shm_cache_collate_identical(tmp_path: Path) -> None:
    """Collating from shm-loaded cache produces the same results as the original."""
    cache = _make_fake_cache(20)
    shm_dir = tmp_path / "fake_shm"
    shm_dir.mkdir()

    try:
        shm_path = save_cache_to_shm(cache, shm_dir=str(shm_dir))
        loaded = load_cache_from_shm(shm_path)

        ds_orig = TensorMapDataset(cache, split_id_keep={0, 1, 2, -1})
        ds_shm = TensorMapDataset(loaded, split_id_keep={0, 1, 2, -1})
        assert len(ds_orig) == len(ds_shm)

        batch_orig = [ds_orig[i] for i in range(min(4, len(ds_orig)))]
        batch_shm = [ds_shm[i] for i in range(min(4, len(ds_shm)))]
        result_orig = compact_collate_fn(batch_orig, cache, include_meta=False)
        result_shm = compact_collate_fn(batch_shm, loaded, include_meta=False)

        for key in ["read_base_idx", "x_num", "label", "mask"]:
            assert torch.equal(result_orig[key], result_shm[key]), f"Collate mismatch: {key}"
    finally:
        shutil.rmtree(shm_dir, ignore_errors=True)
