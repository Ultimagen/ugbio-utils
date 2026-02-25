import torch
from ugbio_srsnv.deep_srsnv.lightning_module import SRSNVLightningModule


def _make_batch(batch_size: int = 4, length: int = 300) -> dict:
    return {
        "read_base_idx": torch.randint(0, 7, (batch_size, length)),
        "ref_base_idx": torch.randint(0, 7, (batch_size, length)),
        "t0_idx": torch.randint(0, 10, (batch_size, length)),
        "x_num": torch.randn(batch_size, 12, length),
        "mask": torch.ones(batch_size, length),
        "label": torch.tensor([1.0, 0.0, 1.0, 0.0][:batch_size]),
        "fold_id": torch.tensor([0, 1, 0, -1][:batch_size]),
    }


def test_lightning_module_forward() -> None:
    model = SRSNVLightningModule(base_vocab_size=8, t0_vocab_size=12, numeric_channels=12)
    batch = _make_batch()
    logits = model._forward(batch)
    assert logits.shape == (4,)


def test_lightning_module_training_step() -> None:
    model = SRSNVLightningModule(base_vocab_size=8, t0_vocab_size=12, numeric_channels=12)
    batch = _make_batch()
    loss = model.training_step(batch, batch_idx=0)
    assert loss.shape == ()
    assert loss.item() > 0


def test_lightning_module_validation_step() -> None:
    model = SRSNVLightningModule(base_vocab_size=8, t0_vocab_size=12, numeric_channels=12)
    batch = _make_batch()
    model.validation_step(batch, batch_idx=0)


def test_lightning_module_predict_step() -> None:
    model = SRSNVLightningModule(base_vocab_size=8, t0_vocab_size=12, numeric_channels=12)
    batch = _make_batch()
    batch["chrom"] = ["chr1", "chr2", "chr1", "chr21"]
    batch["pos"] = torch.tensor([100, 200, 300, 400])
    batch["rn"] = ["r1", "r2", "r3", "r4"]
    result = model.predict_step(batch, batch_idx=0)
    assert "probs" in result
    assert result["probs"].shape == (4,)
    assert all(0.0 <= p <= 1.0 for p in result["probs"].tolist())
    assert result["chrom"] == ["chr1", "chr2", "chr1", "chr21"]
    assert "pos" in result
    assert "rn" in result


def test_lightning_module_hparams_saved() -> None:
    model = SRSNVLightningModule(
        base_vocab_size=8,
        t0_vocab_size=12,
        numeric_channels=12,
        learning_rate=5e-4,
        lr_scheduler="cosine",
    )
    assert model.hparams.learning_rate == 5e-4
    assert model.hparams.lr_scheduler == "cosine"
    assert model.hparams.base_vocab_size == 8


def test_lightning_module_configure_optimizers_no_scheduler() -> None:
    model = SRSNVLightningModule(base_vocab_size=8, t0_vocab_size=12, lr_scheduler="none")
    # Mock trainer with max_epochs
    model._trainer = type("T", (), {"max_epochs": 10, "estimated_stepping_batches": 100})()
    config = model.configure_optimizers()
    assert "optimizer" in config
    assert "lr_scheduler" not in config
