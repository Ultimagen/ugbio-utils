from __future__ import annotations

import functools

import lightning
import torch
from torch.utils.data import DataLoader, Dataset

MAX_BATCH_SIZE = 512


class TensorMapDataset(Dataset):
    """Map-style dataset backed by a shared compact cache via index indirection.

    All instances reference the same underlying tensors in ``full_cache``;
    each split only stores a small int32 index array.
    Returns raw compact tensors from ``__getitem__``; dtype casting and x_num
    expansion happen once per batch in ``compact_collate_fn``.
    """

    def __init__(
        self,
        full_cache: dict,
        split_id_keep: set[int],
        *,
        include_meta: bool = False,
    ):
        self._cache = full_cache
        self.include_meta = include_meta

        split_ids = torch.tensor(sorted(split_id_keep), dtype=torch.long)
        keep_mask = torch.isin(full_cache["split_id"], split_ids)
        self._idx = torch.nonzero(keep_mask, as_tuple=False).flatten().to(dtype=torch.int32)

    def __len__(self) -> int:
        return int(self._idx.shape[0])

    def __getitem__(self, idx: int) -> tuple:
        gi = int(self._idx[idx])
        if self.include_meta:
            return (gi, self._cache["chrom"][gi], self._cache["rn"][gi])
        return (gi,)


def compact_collate_fn(batch: list[tuple], cache: dict, *, include_meta: bool) -> dict:
    """Batch-level collate: index into compact cache and cast once per batch."""
    if include_meta:
        gis, chroms, rns = zip(*batch, strict=False)
    else:
        gis = [b[0] for b in batch]
    idx = torch.tensor(gis, dtype=torch.long)

    x_pos = cache["x_num_pos"][idx].to(dtype=torch.float32)
    x_const = cache["x_num_const"][idx].to(dtype=torch.float32)
    x_num = torch.cat([x_pos, x_const.unsqueeze(-1).expand(-1, -1, x_pos.shape[-1])], dim=1)

    result = {
        "read_base_idx": cache["read_base_idx"][idx].to(dtype=torch.long),
        "ref_base_idx": cache["ref_base_idx"][idx].to(dtype=torch.long),
        "t0_idx": cache["t0_idx"][idx].to(dtype=torch.long),
        "x_num": x_num,
        "mask": cache["mask"][idx].to(dtype=torch.float32),
        "label": cache["label"][idx].to(dtype=torch.float32),
        "fold_id": cache["split_id"][idx].to(dtype=torch.long),
    }
    if include_meta:
        result["chrom"] = list(chroms)
        result["pos"] = torch.from_numpy(cache["pos"][torch.tensor(gis, dtype=torch.long).numpy()]).to(dtype=torch.long)
        result["rn"] = list(rns)
    return result


def _build_loader(
    dataset: TensorMapDataset,
    batch_size: int,
    *,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    include_meta = dataset.include_meta
    cache = dataset._cache
    collate = functools.partial(compact_collate_fn, cache=cache, include_meta=include_meta)
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=bool(pin_memory),
        drop_last=False,
        collate_fn=collate,
    )


class SRSNVDataModule(lightning.LightningDataModule):
    """Lightning DataModule for the deep SRSNV CNN pipeline.

    Wraps the compact tensor cache and provides train/val/test/predict
    DataLoaders. Supports both k-fold and single-model split modes.

    Parameters
    ----------
    full_cache
        The loaded tensor cache dict (from ``load_full_tensor_cache``).
    train_split_ids
        Set of split IDs to include in the training set.
    val_split_ids
        Set of split IDs to include in the validation set.
    test_split_ids
        Set of split IDs for the holdout test set (default ``{-1}``).
    train_batch_size
        Batch size for training.
    eval_batch_size
        Batch size for validation and test.
    predict_batch_size
        Batch size for the prediction/export phase.
    pin_memory
        Whether to pin memory in DataLoaders.
    """

    def __init__(
        self,
        full_cache: dict,
        train_split_ids: set[int],
        val_split_ids: set[int],
        test_split_ids: set[int] | None = None,
        train_batch_size: int = 128,
        eval_batch_size: int | None = None,
        predict_batch_size: int | None = None,
        *,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.full_cache = full_cache
        self.train_split_ids = train_split_ids
        self.val_split_ids = val_split_ids
        self.test_split_ids = test_split_ids or {-1}
        self.train_batch_size = min(train_batch_size, MAX_BATCH_SIZE)
        self.eval_batch_size = min(eval_batch_size or train_batch_size * 2, MAX_BATCH_SIZE * 2)
        self.predict_batch_size = min(predict_batch_size or self.eval_batch_size * 2, MAX_BATCH_SIZE * 4)
        self.pin_memory = pin_memory

        self._train_ds: TensorMapDataset | None = None
        self._val_ds: TensorMapDataset | None = None
        self._test_ds: TensorMapDataset | None = None
        self._predict_ds: TensorMapDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        all_ids = {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        if stage in ("fit", None):
            self._train_ds = TensorMapDataset(self.full_cache, self.train_split_ids)
            self._val_ds = TensorMapDataset(self.full_cache, self.val_split_ids)
        if stage in ("test", None):
            self._test_ds = TensorMapDataset(self.full_cache, self.test_split_ids)
        if stage in ("predict", None):
            self._predict_ds = TensorMapDataset(self.full_cache, all_ids, include_meta=True)

    def train_dataloader(self) -> DataLoader:
        return _build_loader(self._train_ds, self.train_batch_size, shuffle=True, pin_memory=self.pin_memory)

    def val_dataloader(self) -> DataLoader:
        return _build_loader(self._val_ds, self.eval_batch_size, shuffle=False, pin_memory=self.pin_memory)

    def test_dataloader(self) -> DataLoader:
        return _build_loader(self._test_ds, self.eval_batch_size, shuffle=False, pin_memory=self.pin_memory)

    def predict_dataloader(self) -> DataLoader:
        return _build_loader(self._predict_ds, self.predict_batch_size, shuffle=False, pin_memory=self.pin_memory)
