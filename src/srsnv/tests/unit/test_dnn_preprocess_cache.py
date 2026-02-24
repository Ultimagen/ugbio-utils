from pathlib import Path

from ugbio_srsnv.deep_srsnv import data_prep


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x")


def test_build_cached_shards_cache_miss_then_hit(monkeypatch, tmp_path: Path) -> None:
    pos_parquet = tmp_path / "pos.parquet"
    neg_parquet = tmp_path / "neg.parquet"
    pos_bam = tmp_path / "pos.bam"
    neg_bam = tmp_path / "neg.bam"
    for file_path in [pos_parquet, neg_parquet, pos_bam, neg_bam]:
        _touch(file_path)

    shard_rows = [
        [
            {
                "chrom": "chr1",
                "pos": 10,
                "rn": "r1",
                "label": 1,
                "fold_id": 0,
                "read_base_aln": ["A"],
                "ref_base_aln": ["A"],
                "qual_aln": [30.0],
                "tp_aln": [0.0],
                "t0_aln": ["D"],
                "focus_aln": [1.0],
                "softclip_mask_aln": [0.0],
                "strand": 0,
                "mapq": 60.0,
                "rq": 0.5,
                "tm": "AQ",
                "st": "PLUS",
                "et": "PLUS",
                "mixed": 0,
            }
        ],
        [
            {
                "chrom": "chr2",
                "pos": 20,
                "rn": "r2",
                "label": 0,
                "fold_id": 1,
                "read_base_aln": ["C"],
                "ref_base_aln": ["C"],
                "qual_aln": [31.0],
                "tp_aln": [1.0],
                "t0_aln": ["D"],
                "focus_aln": [0.0],
                "softclip_mask_aln": [0.0],
                "strand": 1,
                "mapq": 50.0,
                "rq": 0.4,
                "tm": "AQ",
                "st": "MIXED",
                "et": "PLUS",
                "mixed": 1,
            }
        ],
    ]

    def fake_iter_rows(_path, **kwargs):
        label = kwargs["label"]
        row = {
            "chrom": "chr1" if label else "chr2",
            "pos": 10 if label else 20,
            "ref": "A",
            "alt": "G",
            "X_ALT": "G",
            "RN": "r1" if label else "r2",
            "INDEX": 0,
            "rq": 0.5,
            "tm": "AQ",
            "st": "PLUS",
            "et": "PLUS",
            "label": label,
        }
        yield [row]

    def fake_process_rows_shard(**kwargs):
        sid = kwargs["shard_id"]
        return (
            sid,
            shard_rows[sid],
            {"shard_id": sid, "input_rows": 1, "output_rows": 1, "missing_rows": 0, "wall_seconds": 0.1},
        )

    monkeypatch.setattr(data_prep, "_iter_parquet_rows", fake_iter_rows)
    monkeypatch.setattr(data_prep, "_process_rows_shard", fake_process_rows_shard)

    cache_dir = tmp_path / "cache"
    first = data_prep.build_cached_shards(
        positive_parquet=str(pos_parquet),
        negative_parquet=str(neg_parquet),
        positive_bam=str(pos_bam),
        negative_bam=str(neg_bam),
        chrom_to_fold={"chr1": 0, "chr2": 1},
        split_manifest={"k_folds": 2},
        cache_dir=cache_dir,
        preprocess_num_workers=1,
        preprocess_storage_mode="shards",
    )
    assert first["cache_hit"] is False
    assert len(first["shard_files"]) == 2

    second = data_prep.build_cached_shards(
        positive_parquet=str(pos_parquet),
        negative_parquet=str(neg_parquet),
        positive_bam=str(pos_bam),
        negative_bam=str(neg_bam),
        chrom_to_fold={"chr1": 0, "chr2": 1},
        split_manifest={"k_folds": 2},
        cache_dir=cache_dir,
        preprocess_num_workers=1,
        preprocess_storage_mode="shards",
    )
    assert second["cache_hit"] is True
    assert second["shard_files"] == first["shard_files"]


def test_build_cached_shards_adapts_batch_rows(monkeypatch, tmp_path: Path) -> None:
    pos_parquet = tmp_path / "pos.parquet"
    neg_parquet = tmp_path / "neg.parquet"
    pos_bam = tmp_path / "pos.bam"
    neg_bam = tmp_path / "neg.bam"
    for file_path in [pos_parquet, neg_parquet, pos_bam, neg_bam]:
        _touch(file_path)

    monkeypatch.setattr(data_prep, "_iter_parquet_rows", lambda *_args, **_kwargs: iter([]))
    out = data_prep.build_cached_shards(
        positive_parquet=str(pos_parquet),
        negative_parquet=str(neg_parquet),
        positive_bam=str(pos_bam),
        negative_bam=str(neg_bam),
        chrom_to_fold={},
        split_manifest={"k_folds": 2},
        cache_dir=tmp_path / "cache",
        preprocess_num_workers=16,
        preprocess_max_ram_gb=1.0,
        preprocess_batch_rows=500000,
    )
    assert out["batch_rows"] < 500000


def test_build_cached_shards_mixes_labels_within_split(monkeypatch, tmp_path: Path) -> None:
    pos_parquet = tmp_path / "pos.parquet"
    neg_parquet = tmp_path / "neg.parquet"
    pos_bam = tmp_path / "pos.bam"
    neg_bam = tmp_path / "neg.bam"
    for file_path in [pos_parquet, neg_parquet, pos_bam, neg_bam]:
        _touch(file_path)

    def fake_iter_rows(_path, **kwargs):
        label = kwargs["label"]
        rows = []
        for i in range(4):
            rows.append(
                {
                    "chrom": "chr1",
                    "pos": 100 + i + (0 if label else 10),
                    "ref": "A",
                    "alt": "G",
                    "X_ALT": "G",
                    "RN": f"r_{int(label)}_{i}",
                    "INDEX": i,
                    "rq": 0.5,
                    "tm": "AQ",
                    "st": "PLUS",
                    "et": "PLUS",
                    "label": label,
                }
            )
        yield rows

    seen_labels = []

    def fake_process_rows_shard(**kwargs):
        sid = kwargs["shard_id"]
        rows = kwargs["rows"]
        seen_labels.append([bool(r["label"]) for r in rows])
        return (
            sid,
            [],
            {"shard_id": sid, "input_rows": len(rows), "output_rows": 0, "missing_rows": 0, "wall_seconds": 0.1},
        )

    monkeypatch.setattr(data_prep, "_iter_parquet_rows", fake_iter_rows)
    monkeypatch.setattr(data_prep, "_process_rows_shard", fake_process_rows_shard)

    out = data_prep.build_cached_shards(
        positive_parquet=str(pos_parquet),
        negative_parquet=str(neg_parquet),
        positive_bam=str(pos_bam),
        negative_bam=str(neg_bam),
        chrom_to_fold={"chr1": 0},
        split_manifest={"k_folds": 2, "random_seed": 42},
        cache_dir=tmp_path / "cache",
        preprocess_num_workers=1,
        preprocess_batch_rows=16,
    )
    assert out["cache_hit"] is False
    assert "split_input_stats" in out
    # At least one shard should contain both classes after balancing/shuffle.
    assert any(any(labels) and not all(labels) for labels in seen_labels)
