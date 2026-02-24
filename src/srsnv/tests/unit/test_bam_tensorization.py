import numpy as np
from ugbio_srsnv.deep_srsnv.data_prep import DeepSRSNVDataset, build_encoders


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
            "t0_aln": ["D", "D", ":", "<GAP>", "D", "-"],
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
    encoders = build_encoders(records)
    ds = DeepSRSNVDataset(records, encoders=encoders, length=300)
    item = ds[0]
    assert item["read_base_idx"].shape[0] == 300
    assert item["ref_base_idx"].shape[0] == 300
    assert item["t0_idx"].shape[0] == 300
    assert item["x_num"].shape == (12, 300)
    assert item["mask"].shape[0] == 300
    assert item["label"].item() == 1.0
    # focus one-hot is preserved before pad.
    assert float(item["x_num"][3, :6].sum().item()) == 1.0
    # softclip mask channel index=4 is preserved before pad.
    assert np.allclose(item["x_num"][4, :6].numpy(), np.array([0, 1, 1, 0, 0, 0], dtype=np.float32))
