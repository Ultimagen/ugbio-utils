import json
from pathlib import Path

import numpy as np
import pytest
from ugbio_srsnv import srsnv_dnn_bam_training as dnn_train


@pytest.mark.integration
def test_dnn_training_pipeline_with_mocked_data(monkeypatch, tmp_path: Path) -> None:  # noqa: C901
    resources = Path(__file__).parent.parent / "resources"
    interval_list = resources / "wgs_calling_regions.without_encode_blacklist.hg38.interval_list.gz"

    def _fake_schema(*_args, **_kwargs):
        return {"schema_version": 1, "tag_counts": {"tp": 10, "t0": 10}}

    def _fake_records(*_args, **_kwargs):
        records = []
        chroms = ["chr1", "chr2", "chr21", "chr22"]
        for i in range(80):
            chrom = chroms[i % len(chroms)]
            fold = -1 if chrom in {"chr21", "chr22"} else i % 2
            records.append(
                {
                    "chrom": chrom,
                    "pos": 1000 + i,
                    "ref": "A",
                    "alt": "G",
                    "rn": f"read_{i}",
                    "label": int((i % 3) == 0),
                    "fold_id": fold,
                    "read_base_aln": list("ACGT" * 40),
                    "ref_base_aln": list("ACGT" * 40),
                    "qual_aln": np.full(160, 30.0, dtype=np.float32),
                    "tp_aln": np.zeros(160, dtype=np.float32),
                    "t0_aln": ["D"] * 160,
                    "focus_aln": np.array([1.0 if j == 80 else 0.0 for j in range(160)], dtype=np.float32),
                    "softclip_mask_aln": np.array([1.0 if j < 5 else 0.0 for j in range(160)], dtype=np.float32),
                    "strand": i % 2,
                    "mapq": 60.0,
                    "rq": 0.5,
                    "tm": "AQ",
                    "st": "MIXED" if i % 2 == 0 else "PLUS",
                    "et": "PLUS" if i % 2 == 0 else "MINUS",
                    "mixed": int(i % 2 == 0),
                    "index": i % 100,
                    "read_len": 160,
                }
            )
        return records

    fake_shards = [str(tmp_path / "shard_00000.pkl"), str(tmp_path / "shard_00001.pkl")]
    shard_a = _fake_records()[:40]
    shard_b = _fake_records()[40:]

    def _fake_build_cached_shards(*_args, **_kwargs):
        return {
            "cache_key": "test",
            "cache_hit": False,
            "shard_files": fake_shards,
            "records_stream_path": str(tmp_path / "records_stream.pkl"),
            "total_shards": 2,
            "total_output_rows": 80,
            "peak_rss_gb": 1.0,
        }

    def _fake_iter_cached_records(shard_files=None, records_stream_path=None):
        del records_stream_path
        for shard in shard_files:
            if shard == fake_shards[0]:
                yield shard_a
            else:
                yield shard_b

    tensor_chunks = []

    def _fake_build_tensor_cache_from_records_stream(*_args, **_kwargs):
        nonlocal tensor_chunks
        tensor_chunks = []
        for shard in [shard_a, shard_b]:
            chunk = {
                "read_base_idx": np.zeros((len(shard), 300), dtype=np.int64),
                "ref_base_idx": np.zeros((len(shard), 300), dtype=np.int64),
                "t0_idx": np.zeros((len(shard), 300), dtype=np.int64),
                "x_num": np.zeros((len(shard), 12, 300), dtype=np.float32),
                "mask": np.ones((len(shard), 300), dtype=np.float32),
                "label": np.array([r["label"] for r in shard], dtype=np.float32),
                "split_id": np.array([r["fold_id"] for r in shard], dtype=np.int64),
                "chrom": np.array([r["chrom"] for r in shard], dtype=object),
                "pos": np.array([r["pos"] for r in shard], dtype=np.int64),
                "rn": np.array([r["rn"] for r in shard], dtype=object),
            }
            tensor_chunks.append(chunk)
            return {
                "tensor_cache_path": str(tmp_path / "tensor_cache.pkl"),
                "tensor_cache_chunks": 2,
                "tensor_cache_rows": 80,
                "split_counts": {
                    "0": {"rows": 20, "positives": 7, "negatives": 13},
                    "1": {"rows": 20, "positives": 7, "negatives": 13},
                    "-1": {"rows": 40, "positives": 12, "negatives": 28},
                },
                "chunk_split_stats": [
                    {
                        "chunk_id": 1,
                        "split_stats": {"0": {"rows": 20, "positives": 7, "negatives": 13, "prevalence": 0.35}},
                    },
                    {
                        "chunk_id": 2,
                        "split_stats": {"0": {"rows": 20, "positives": 7, "negatives": 13, "prevalence": 0.35}},
                    },
                ],
            }

    def _fake_iter_tensor_cache_chunks(_tensor_cache_path):
        yield from tensor_chunks

    monkeypatch.setattr(dnn_train, "discover_bam_schema", _fake_schema)
    monkeypatch.setattr(dnn_train, "build_cached_shards", _fake_build_cached_shards)
    monkeypatch.setattr(dnn_train, "iter_cached_records", _fake_iter_cached_records)
    monkeypatch.setattr(
        dnn_train, "build_tensor_cache_from_records_stream", _fake_build_tensor_cache_from_records_stream
    )
    monkeypatch.setattr(dnn_train, "iter_tensor_cache_chunks", _fake_iter_tensor_cache_chunks)
    fake_args = type(
        "Args",
        (),
        {
            "positive_bam": "ignored.bam",
            "negative_bam": "ignored.bam",
            "positive_parquet": "ignored.parquet",
            "negative_parquet": "ignored.parquet",
            "training_regions": str(interval_list),
            "output": str(tmp_path),
            "basename": "dnn_test",
            "k_folds": 2,
            "split_manifest_in": None,
            "split_manifest_out": str(tmp_path / "split.json"),
            "holdout_chromosomes": "chr21,chr22",
            "single_model_split": False,
            "val_fraction": 0.1,
            "split_hash_key": "RN",
            "max_rows_per_class": None,
            "epochs": 1,
            "batch_size": 16,
            "learning_rate": 1e-3,
            "random_seed": 1,
            "length": 300,
            "preprocess_cache_dir": str(tmp_path / "cache"),
            "preprocess_num_workers": 1,
            "preprocess_max_ram_gb": 8.0,
            "preprocess_batch_rows": 1024,
            "preprocess_storage_mode": "single_file",
            "encoder_vocab_source": "known",
            "preprocess_dry_run": False,
            "loader_num_workers": 0,
            "loader_prefetch_factor": 2,
            "loader_pin_memory": False,
            "use_amp": False,
            "use_tf32": False,
            "autotune_batch_size": False,
            "gpu_telemetry_interval_steps": 50,
            "verbose": False,
        },
    )()
    monkeypatch.setattr(dnn_train, "_cli", lambda: fake_args)

    dnn_train.main()
    assert (tmp_path / "dnn_test.featuremap_df.parquet").is_file()
    assert (tmp_path / "dnn_test.srsnv_dnn_metadata.json").is_file()
    meta = json.loads((tmp_path / "dnn_test.srsnv_dnn_metadata.json").read_text())
    assert "training_results" in meta
    assert "training_runtime_metrics" in meta
    assert "split_prevalence" in meta
    assert "chunk_composition" in meta
    assert "model_architecture" in meta
    assert "split_manifest" in meta
    assert "st_vocab" in meta["encoders"]
    assert "et_vocab" in meta["encoders"]
    assert "focus" in meta["channel_order"]
    assert "softclip_mask" in meta["channel_order"]
    assert "aupr" in meta["holdout_metrics"]
