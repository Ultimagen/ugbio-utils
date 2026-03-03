"""Callback that evaluates the SWA averaged model on the validation set each epoch.

Lightning's built-in ``StochasticWeightAveraging`` callback maintains an
``_average_model`` shadow copy but never evaluates it during training.  This
callback fills that gap: after each validation epoch during the SWA phase it
swaps the averaged weights into the model, refits BatchNorm statistics on a
subset of the training data, runs a validation forward pass, and logs
``swa_val_auc`` / ``swa_val_aupr`` alongside the normal training metrics.
"""

from __future__ import annotations

from typing import Any

import torch
import torchmetrics
from lightning.pytorch.callbacks import Callback, StochasticWeightAveraging
from torch import nn
from ugbio_core.logger import logger

_BN_REFIT_MAX_BATCHES = 200


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class SWAValidationTracker(Callback):
    """Track validation metrics of the SWA averaged model during training.

    Parameters
    ----------
    swa_callback
        The ``StochasticWeightAveraging`` callback instance whose
        ``_average_model`` will be evaluated.
    bn_refit_max_batches
        Maximum number of training batches used to refit BatchNorm running
        statistics before validation.  Set to 0 to skip the BN refit (metrics
        will be approximate).
    """

    def __init__(
        self,
        swa_callback: StochasticWeightAveraging,
        bn_refit_max_batches: int = _BN_REFIT_MAX_BATCHES,
    ):
        super().__init__()
        self._swa_cb = swa_callback
        self._bn_refit_max_batches = bn_refit_max_batches
        self._auroc = torchmetrics.AUROC(task="binary")
        self._ap = torchmetrics.AveragePrecision(task="binary")

    def _is_swa_active(self, trainer) -> bool:
        return (
            self._swa_cb._initialized
            and self._swa_cb._average_model is not None
            and self._swa_cb.n_averaged is not None
            and self._swa_cb.n_averaged.item() > 0
            and self._swa_cb.swa_start <= trainer.current_epoch <= self._swa_cb.swa_end
        )

    @staticmethod
    def _save_state(pl_module) -> dict[str, Any]:
        """Snapshot all parameters and BN buffers so we can restore them."""
        return {
            "params": [p.data.clone() for p in pl_module.parameters()],
            "bn_buffers": [
                (m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
                for m in pl_module.modules()
                if isinstance(m, nn.modules.batchnorm._BatchNorm)
            ],
        }

    @staticmethod
    def _restore_state(pl_module, state: dict[str, Any]) -> None:
        for p, saved in zip(pl_module.parameters(), state["params"], strict=True):
            p.data.copy_(saved)
        bn_idx = 0
        for m in pl_module.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                mean, var, tracked = state["bn_buffers"][bn_idx]
                m.running_mean.copy_(mean)
                m.running_var.copy_(var)
                m.num_batches_tracked.copy_(tracked)
                bn_idx += 1

    @staticmethod
    def _copy_averaged_weights(avg_model, pl_module) -> None:
        for p_avg, p_dst in zip(avg_model.parameters(), pl_module.parameters(), strict=True):
            p_dst.data.copy_(p_avg.data.to(p_dst.device))

    def _refit_bn(self, pl_module, trainer) -> None:
        """Run a forward pass over training data to recompute BN running stats."""
        has_bn = any(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in pl_module.modules())
        if not has_bn or self._bn_refit_max_batches <= 0:
            return

        for m in pl_module.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.running_mean.zero_()
                m.running_var.fill_(1.0)
                m.num_batches_tracked.zero_()
                m.momentum = None  # use cumulative mean

        pl_module.train()
        train_dl = trainer.datamodule.train_dataloader()
        with torch.no_grad():
            for i, batch in enumerate(train_dl):
                if i >= self._bn_refit_max_batches:
                    break
                pl_module._forward(_batch_to_device(batch, pl_module.device))

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self._is_swa_active(trainer):
            return

        state = self._save_state(pl_module)

        try:
            self._copy_averaged_weights(self._swa_cb._average_model, pl_module)
            self._refit_bn(pl_module, trainer)
            pl_module.eval()

            self._auroc.to(pl_module.device)
            self._ap.to(pl_module.device)

            val_dl = trainer.datamodule.val_dataloader()
            with torch.no_grad():
                for batch in val_dl:
                    moved = _batch_to_device(batch, pl_module.device)
                    logits = pl_module._forward(moved)
                    preds = torch.sigmoid(logits)
                    labels = moved["label"].int()
                    self._auroc.update(preds, labels)
                    self._ap.update(preds, labels)

            swa_auc = self._auroc.compute()
            swa_aupr = self._ap.compute()
            self._auroc.reset()
            self._ap.reset()

            pl_module.log("swa_val_auc", swa_auc, prog_bar=True, sync_dist=True)
            pl_module.log("swa_val_aupr", swa_aupr, prog_bar=True, sync_dist=True)
            logger.info(
                "SWA validation (epoch %d, n_averaged=%d): auc=%.6f aupr=%.6f",
                trainer.current_epoch,
                self._swa_cb.n_averaged.item(),
                swa_auc.item(),
                swa_aupr.item(),
            )
        finally:
            self._restore_state(pl_module, state)
