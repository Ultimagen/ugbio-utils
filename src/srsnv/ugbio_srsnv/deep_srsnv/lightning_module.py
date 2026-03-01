from __future__ import annotations

from typing import Any

import lightning
import torch
import torchmetrics
from torch import nn

from ugbio_srsnv.deep_srsnv.cnn_model import CNNReadClassifier

LR_SCHEDULER_CHOICES = ("none", "cosine", "step", "onecycle", "reduce_on_plateau")


class SRSNVLightningModule(lightning.LightningModule):
    """PyTorch Lightning wrapper for CNNReadClassifier.

    Handles training/validation/test/predict steps, optimizer and LR scheduler
    configuration, and metric logging.

    Parameters
    ----------
    base_vocab_size
        Size of the base vocabulary for the embedding layer.
    t0_vocab_size
        Size of the t0 vocabulary for the embedding layer.
    numeric_channels
        Number of numeric input channels.
    learning_rate
        Peak learning rate for the optimizer.
    weight_decay
        Weight decay for AdamW optimizer.
    lr_scheduler
        LR scheduler type. One of ``LR_SCHEDULER_CHOICES``.
    lr_warmup_epochs
        Number of warmup epochs (used by ``onecycle``).
    lr_min
        Minimum learning rate (used by ``cosine``).
    lr_step_size
        Step size for ``step`` scheduler (epochs between LR drops).
    lr_gamma
        Multiplicative decay factor for ``step`` / ``reduce_on_plateau``.
    lr_patience
        Patience for ``reduce_on_plateau`` scheduler.
    """

    def __init__(  # noqa: PLR0913
        self,
        base_vocab_size: int,
        t0_vocab_size: int,
        numeric_channels: int = 9,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lr_scheduler: str = "onecycle",
        lr_warmup_epochs: int = 1,
        lr_min: float = 1e-6,
        lr_step_size: int = 5,
        lr_gamma: float = 0.5,
        lr_patience: int = 3,
        **model_kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = CNNReadClassifier(
            base_vocab_size=base_vocab_size,
            t0_vocab_size=t0_vocab_size,
            numeric_channels=numeric_channels,
            **model_kwargs,
        )
        self.criterion = nn.BCEWithLogitsLoss()

        # sync_on_compute=False: each rank computes metrics on its local data only.
        # This avoids NCCL all_gather calls that can desync with DDP collectives.
        self.train_auroc = torchmetrics.AUROC(task="binary", sync_on_compute=False)
        self.train_ap = torchmetrics.AveragePrecision(task="binary", sync_on_compute=False)
        self.val_auroc = torchmetrics.AUROC(task="binary", sync_on_compute=False)
        self.val_ap = torchmetrics.AveragePrecision(task="binary", sync_on_compute=False)
        self.test_auroc = torchmetrics.AUROC(task="binary", sync_on_compute=False)
        self.test_ap = torchmetrics.AveragePrecision(task="binary", sync_on_compute=False)

    def _forward(self, batch: dict) -> torch.Tensor:
        return self.model(
            read_base_idx=batch["read_base_idx"],
            ref_base_idx=batch["ref_base_idx"],
            t0_idx=batch["t0_idx"],
            x_num=batch["x_num"],
            mask=batch["mask"],
            tm_idx=batch.get("tm_idx"),
            st_idx=batch.get("st_idx"),
            et_idx=batch.get("et_idx"),
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        logits = self._forward(batch)
        loss = self.criterion(logits, batch["label"])
        preds = torch.sigmoid(logits)
        labels = batch["label"].int()

        self.train_auroc.update(preds, labels)
        self.train_ap.update(preds, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_train_epoch_end(self) -> None:
        # sync_on_compute=False disables the torchmetrics all_gather, so
        # sync_dist=True here is the single collective that ensures all
        # ranks see identical metric values (needed for EarlyStopping).
        self.log("train_auc", self.train_auroc.compute(), prog_bar=True, sync_dist=True)
        self.log("train_aupr", self.train_ap.compute(), prog_bar=True, sync_dist=True)
        self.train_auroc.reset()
        self.train_ap.reset()

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        logits = self._forward(batch)
        loss = self.criterion(logits, batch["label"])
        preds = torch.sigmoid(logits)
        labels = batch["label"].int()

        self.val_auroc.update(preds, labels)
        self.val_ap.update(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val_auc", self.val_auroc.compute(), prog_bar=True, sync_dist=True)
        self.log("val_aupr", self.val_ap.compute(), prog_bar=True, sync_dist=True)
        self.val_auroc.reset()
        self.val_ap.reset()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        logits = self._forward(batch)
        loss = self.criterion(logits, batch["label"])
        preds = torch.sigmoid(logits)
        labels = batch["label"].int()

        self.test_auroc.update(preds, labels)
        self.test_ap.update(preds, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

    def on_test_epoch_end(self) -> None:
        self.log("test_auc", self.test_auroc.compute(), sync_dist=True)
        self.log("test_aupr", self.test_ap.compute(), sync_dist=True)
        self.test_auroc.reset()
        self.test_ap.reset()

    def predict_step(self, batch: dict, batch_idx: int) -> dict:
        logits = self._forward(batch)
        probs = torch.sigmoid(logits)
        result: dict[str, Any] = {"probs": probs, "label": batch["label"], "fold_id": batch["fold_id"]}
        if "chrom" in batch:
            result["chrom"] = batch["chrom"]
        if "pos" in batch:
            result["pos"] = batch["pos"]
        if "rn" in batch:
            result["rn"] = batch["rn"]
        return result

    def configure_optimizers(self) -> dict:
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "bn" in name or "bias" in name or "_emb" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.hparams.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.learning_rate,
        )
        config: dict[str, Any] = {"optimizer": optimizer}

        scheduler_name = self.hparams.lr_scheduler
        if scheduler_name == "none":
            return config

        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.hparams.lr_min,
            )
            config["lr_scheduler"] = {"scheduler": scheduler, "interval": "epoch"}
        elif scheduler_name == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.hparams.lr_step_size,
                gamma=self.hparams.lr_gamma,
            )
            config["lr_scheduler"] = {"scheduler": scheduler, "interval": "epoch"}
        elif scheduler_name == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=min(self.hparams.lr_warmup_epochs / max(1, self.trainer.max_epochs), 0.3),
            )
            config["lr_scheduler"] = {"scheduler": scheduler, "interval": "step"}
        elif scheduler_name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=self.hparams.lr_gamma,
                patience=self.hparams.lr_patience,
                min_lr=self.hparams.lr_min,
            )
            config["lr_scheduler"] = {"scheduler": scheduler, "monitor": "val_auc", "interval": "epoch"}

        return config
