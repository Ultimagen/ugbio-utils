from __future__ import annotations

import functools
import os
from pathlib import Path

import lightning
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from ugbio_core.logger import logger

from ugbio_srsnv.deep_srsnv.data_prep import compute_split_ids


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
        split_id_keep: set[int] | None = None,
        *,
        include_meta: bool = False,
    ):
        self._cache = full_cache
        self.include_meta = include_meta

        if split_id_keep is not None and "split_id" in full_cache:
            split_ids = torch.tensor(sorted(split_id_keep), dtype=torch.long)
            keep_mask = torch.isin(full_cache["split_id"], split_ids)
            self._idx = torch.nonzero(keep_mask, as_tuple=False).flatten().to(dtype=torch.int32)
        else:
            n = int(full_cache["label"].shape[0])
            self._idx = torch.arange(n, dtype=torch.int32)

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
        "x_num": x_num,
        "mask": cache["mask"][idx].to(dtype=torch.float32),
        "label": cache["label"][idx].to(dtype=torch.float32),
        "fold_id": cache["split_id"][idx].to(dtype=torch.long),
    }
    if "tm_idx" in cache:
        result["tm_idx"] = cache["tm_idx"][idx].to(dtype=torch.long)
        result["st_idx"] = cache["st_idx"][idx].to(dtype=torch.long)
        result["et_idx"] = cache["et_idx"][idx].to(dtype=torch.long)
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
    num_workers: int = 0,
    prefetch_factor: int | None = None,
) -> DataLoader:
    include_meta = dataset.include_meta
    cache = dataset._cache
    collate = functools.partial(compact_collate_fn, cache=cache, include_meta=include_meta)
    pf = prefetch_factor if num_workers > 0 else None
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=num_workers,
        prefetch_factor=pf,
        pin_memory=bool(pin_memory),
        drop_last=False,
        collate_fn=collate,
    )


_DEFAULT_LOADER_WORKERS = max(1, min((os.cpu_count() or 4) // 4, 8))


def _load_fold_file(path: Path, split_id_value: int = 0, *, mmap: bool = False) -> dict:
    """Load a single fold file (train.pt, val.pt, or test.pt).

    Injects a ``split_id`` tensor so that ``compact_collate_fn`` works
    identically to the legacy path.
    When ``mmap=True``, tensors are memory-mapped so DDP processes share
    physical RAM pages.
    """
    cache = dict(torch.load(path, map_location="cpu", weights_only=False, mmap=mmap))  # noqa: S301
    if isinstance(cache.get("chrom"), np.ndarray):
        cache["chrom"] = cache["chrom"].tolist()
    if isinstance(cache.get("rn"), np.ndarray):
        cache["rn"] = cache["rn"].tolist()
    if "split_id" not in cache:
        n = int(cache["label"].shape[0])
        cache["split_id"] = torch.full((n,), split_id_value, dtype=torch.int8)
    return cache


class SRSNVDataModule(lightning.LightningDataModule):
    """Lightning DataModule for the deep SRSNV CNN pipeline.

    Supports two modes:

    1. **Legacy (split-at-runtime)**: pass ``full_cache`` + ``train_split_ids``
       + ``val_split_ids``. Split IDs are computed from CHROM at init time.

    2. **Fold-directory (pre-split)**: use ``SRSNVDataModule.from_fold_dir()``
       to load pre-split ``train.pt`` / ``val.pt`` / ``test.pt`` files written
       by ``combine_and_split``. No split computation needed.
    """

    def __init__(  # noqa: PLR0913
        self,
        full_cache: dict | None = None,
        train_split_ids: set[int] | None = None,
        val_split_ids: set[int] | None = None,
        test_split_ids: set[int] | None = None,
        train_batch_size: int = 128,
        eval_batch_size: int | None = None,
        predict_batch_size: int | None = None,
        *,
        pin_memory: bool = False,
        split_manifest: dict | None = None,
        chrom_to_fold: dict[str, int] | None = None,
        num_workers: int = 0,
        prefetch_factor: int = 4,
        fold_dir: str | Path | None = None,
        use_mmap: bool = False,
    ):
        super().__init__()

        self._fold_dir = Path(fold_dir) if fold_dir else None
        self._fold_caches: dict[str, dict] | None = None

        if self._fold_dir is not None:
            self._fold_caches = {}
            split_id_map = {"train": 0, "val": 1, "test": -1}
            for name in ("train", "val", "test"):
                fp = self._fold_dir / f"{name}.pt"
                if fp.exists():
                    self._fold_caches[name] = _load_fold_file(fp, split_id_value=split_id_map[name], mmap=use_mmap)
                    logger.info("Loaded %s: %d rows", fp, int(self._fold_caches[name]["label"].shape[0]))
            self.full_cache = None
            self.train_split_ids = None
            self.val_split_ids = None
            self.test_split_ids = None
        else:
            if full_cache is None:
                raise ValueError("Either full_cache or fold_dir must be provided")
            self._ensure_split_id(full_cache, split_manifest, chrom_to_fold)
            self.full_cache = full_cache
            self.train_split_ids = train_split_ids or set()
            self.val_split_ids = val_split_ids or set()
            self.test_split_ids = test_split_ids or {-1}

        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size or train_batch_size * 2
        self.predict_batch_size = predict_batch_size or self.eval_batch_size * 2
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self._train_ds: TensorMapDataset | None = None
        self._val_ds: TensorMapDataset | None = None
        self._test_ds: TensorMapDataset | None = None
        self._predict_ds: TensorMapDataset | None = None

    @classmethod
    def from_fold_dir(
        cls,
        fold_dir: str | Path,
        train_batch_size: int = 128,
        eval_batch_size: int | None = None,
        predict_batch_size: int | None = None,
        *,
        pin_memory: bool = False,
        num_workers: int = 0,
        prefetch_factor: int = 4,
        use_mmap: bool = False,
    ) -> SRSNVDataModule:
        """Create a DataModule from a pre-split fold directory."""
        return cls(
            fold_dir=fold_dir,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            predict_batch_size=predict_batch_size,
            pin_memory=pin_memory,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            use_mmap=use_mmap,
        )

    @staticmethod
    def _ensure_split_id(
        full_cache: dict,
        split_manifest: dict | None,
        chrom_to_fold: dict[str, int] | None,
    ) -> None:
        if split_manifest is not None or chrom_to_fold is not None:
            chroms = full_cache["chrom"]
            rns = full_cache.get("rn")
            full_cache["split_id"] = compute_split_ids(
                chroms=chroms,
                rns=rns,
                split_manifest=split_manifest,
                chrom_to_fold=chrom_to_fold or {},
            )
            n_rows = len(chroms)
            unique_splits = sorted(set(full_cache["split_id"].tolist()))
            logger.info(
                "Computed split_id from CHROM: %d rows, splits=%s",
                n_rows,
                ",".join(str(s) for s in unique_splits),
            )
        elif "split_id" not in full_cache:
            n_rows = len(full_cache["chrom"])
            full_cache["split_id"] = torch.zeros(n_rows, dtype=torch.int8)
            logger.info("No split info provided — defaulting all %d rows to split_id=0", n_rows)

    def setup(self, stage: str | None = None) -> None:
        if self._fold_caches is not None:
            if stage in ("fit", None):
                if "train" in self._fold_caches:
                    self._train_ds = TensorMapDataset(self._fold_caches["train"])
                if "val" in self._fold_caches:
                    self._val_ds = TensorMapDataset(self._fold_caches["val"])
            if stage in ("test", None) and "test" in self._fold_caches:
                self._test_ds = TensorMapDataset(self._fold_caches["test"])
            if stage in ("predict", None):
                all_data = self._merge_fold_caches_for_predict()
                self._predict_ds = TensorMapDataset(all_data, include_meta=True)
            return

        all_ids = {-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
        if stage in ("fit", None):
            self._train_ds = TensorMapDataset(self.full_cache, self.train_split_ids)
            self._val_ds = TensorMapDataset(self.full_cache, self.val_split_ids)
        if stage in ("test", None):
            self._test_ds = TensorMapDataset(self.full_cache, self.test_split_ids)
        if stage in ("predict", None):
            self._predict_ds = TensorMapDataset(self.full_cache, all_ids, include_meta=True)

    def _merge_fold_caches_for_predict(self) -> dict:
        """Merge val/test caches into a single cache for predict phase.

        Training data is excluded since predictions on it are not useful
        for evaluation and would waste compute.
        """
        caches_to_merge = [self._fold_caches[k] for k in ("val", "test") if k in self._fold_caches]
        if len(caches_to_merge) == 1:
            return caches_to_merge[0]

        merged: dict = {}
        for key in (
            "read_base_idx",
            "ref_base_idx",
            "tm_idx",
            "st_idx",
            "et_idx",
            "x_num_pos",
            "x_num_const",
            "mask",
            "label",
        ):
            tensors = [c[key] for c in caches_to_merge if key in c]
            if tensors:
                merged[key] = torch.cat(tensors, dim=0)

        merged["chrom"] = []
        merged["rn"] = []
        pos_parts = []
        for c in caches_to_merge:
            merged["chrom"].extend(c.get("chrom", []))
            merged["rn"].extend(c.get("rn", []))
            pos_parts.append(np.asarray(c.get("pos", []), dtype=np.int32))
        merged["pos"] = np.concatenate(pos_parts) if pos_parts else np.array([], dtype=np.int32)

        fold_ids = []
        for split_name in ("val", "test"):
            if split_name not in self._fold_caches:
                continue
            cn = int(self._fold_caches[split_name]["label"].shape[0])
            sid = {"val": 1, "test": -1}[split_name]
            fold_ids.extend([sid] * cn)
        merged["split_id"] = torch.tensor(fold_ids, dtype=torch.int8)

        return merged

    def train_dataloader(self) -> DataLoader:
        return _build_loader(
            self._train_ds,
            self.train_batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self) -> DataLoader:
        return _build_loader(
            self._val_ds,
            self.eval_batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self) -> DataLoader:
        return _build_loader(
            self._test_ds,
            self.eval_batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def predict_dataloader(self) -> DataLoader:
        return _build_loader(
            self._predict_ds,
            self.predict_batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )
