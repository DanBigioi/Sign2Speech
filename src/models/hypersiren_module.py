#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
HyperNetwork predicting the weights of a SIREN from an embedding of the hand pose.
"""

import pytorch_lightning as pl
import torch

from typing import Any, List
from torchmetrics import MinMetric
from torchmetrics.regression.mse import MeanSquaredError

from src.models.components.hypernetwork import HyperNetwork


class HyperSirenLitModule(pl.LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        hyponet: torch.nn.Module,
        latent_dim: int,
        hyper_hidden_layers: int = 1,
        hyper_hidden_features: int = 256,
        lr: float = 0.001,
        weight_decay: float = 0.00005,
    ):
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.encoder = encoder
        self.hyponet = hyponet
        self.hypernet = HyperNetwork(
            hyper_in_features=latent_dim,  # The output dimensionality of the encoder
            hyper_hidden_layers=hyper_hidden_layers,
            hyper_hidden_features=hyper_hidden_features,
            hypo_module=self.hyponet,
        )
        # loss function is MSE for audio
        self.criterion = torch.nn.MSELoss()
        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_loss = MeanSquaredError()
        self.val_loss = MeanSquaredError()
        self.test_loss = MeanSquaredError()
        # for logging best so far validation loss
        self.val_loss_best = MinMetric()

    def freeze_hypernet(self):
        for p in self.hypernet.parameters():
            p.requires_grad = False

    def get_hypo_net_weights(self, x: torch.Tensor):
        """
        The model input is a hand pose. This method returns the predicted weights for the
        hypo-network.
        """
        embedding = self.encoder(x)
        hypo_params = self.hypernet(embedding)
        return hypo_params, embedding

    def step(self, batch: Any):
        x, y, l = batch
        loss = .0
        for _ in range(100):
            hypo_params = self.hypernet(self.encoder(x['pose']))
            y_hat = self.hyponet(x['audio'], params=hypo_params)
            loss += self.criterion(y_hat, y['amplitude'])
        return loss, y_hat, y

    def training_step(self, train_batch, batch_idx):
        loss, y_hat, y = self.step(train_batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
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
        # TODO: SignalToNoiseRatio or other audio metrics?
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
