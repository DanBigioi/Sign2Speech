# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
LSTM-based models.
"""

import torch
import pytorch_lightning as pl

from typing import Any, List
from torchmetrics import MinMetric
from torchmetrics.regression.mse import MeanSquaredError


class LSTMLitModule(pl.LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.00005,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.MSELoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_loss = MeanSquaredError()
        self.val_loss = MeanSquaredError()
        self.test_loss = MeanSquaredError()

        # for logging best so far validation accuracy
        self.val_loss_best = MinMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x, y, l = batch
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(self, train_batch, batch_idx):
        loss, y_hat, y = self.step(train_batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # TODO: SignalToNoiseRatio or other audio metrics once we have incorporated a neural
        # vocoder

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": y_hat, "targets": y}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, val_batch, batch_idx):
        loss, y_hat, y = self.step(val_batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": y_hat, "targets": y}

    def validation_epoch_end(self, outputs: List[Any]):
        loss = self.val_loss.compute()
        self.val_loss_best.update(loss)
        self.log(
            "val/loss_best", self.val_loss_best.compute(), on_epoch=True, prog_bar=True
        )

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        # log test metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_loss.reset()
        self.test_loss.reset()
        self.val_loss.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
