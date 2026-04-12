import numpy as np
from ugbio_srsnv.deep_srsnv.data_prep import DeepSRSNVDataset, load_vocab_config


def test_tensorization_shapes() -> None:
    records = [
        {
            "chrom": "chr1",
            "pos": 100,
            "ref": "A",
            "alt": "G",
            "rn": "r1",
            "label": 1,
            "fold_id": 0,
            "read_base_aln": ["A", "C", "T", "<GAP>", "A", "C"],
            "ref_base_aln": ["A", "C", "<GAP>", "T", "A", "C"],
            "qual_aln": np.array([30, 31, 32, 0, 34, 35], dtype=np.float32),
            "tp_aln": np.array([0, 1, 0, 0, 1, 0], dtype=np.float32),
            "t0_aln": np.array([35, 35, 25, 0, 35, 12], dtype=np.float32),
            "focus_aln": np.array([0, 0, 0, 1, 0, 0], dtype=np.float32),
            "softclip_mask_aln": np.array([0, 1, 1, 0, 0, 0], dtype=np.float32),
            "strand": 1,
            "mapq": 60.0,
            "rq": 0.4,
            "tm": "AQ",
            "st": "MIXED",
            "et": "PLUS",
            "mixed": 1,
            "index": 3,
            "read_len": 6,
        }
    ]
    encoders = load_vocab_config()
    ds = DeepSRSNVDataset(records, encoders=encoders, length=300)
    item = ds[0]
    assert item["read_base_idx"].shape[0] == 300
    assert item["ref_base_idx"].shape[0] == 300
    assert item["x_num"].shape == (10, 300)
    assert item["mask"].shape[0] == 300
    assert item["label"].item() == 1.0
    assert "tm_idx" in item
    assert "st_idx" in item
    assert "et_idx" in item
    # focus one-hot is preserved before pad.
    assert float(item["x_num"][3, :6].sum().item()) == 1.0
    # softclip mask channel index=4 is preserved before pad.
    assert np.allclose(item["x_num"][4, :6].numpy(), np.array([0, 1, 1, 0, 0, 0], dtype=np.float32))
    # t0 numeric channel index=5 is preserved before pad.
    expected_t0 = np.array([35, 35, 25, 0, 35, 12], dtype=np.float32) / 10.0
    assert np.allclose(item["x_num"][5, :6].numpy(), expected_t0)
