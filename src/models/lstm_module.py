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
        # Workaround to load model mapped on GPU
        # https://stackoverflow.com/a/61840832
        # waveglow = torch.hub.load(
        #     "NVIDIA/DeepLearningExamples:torchhub",
        #     "nvidia_waveglow",
        #     model_math="fp32",
        #     pretrained=False,
        # )
        # checkpoint = torch.hub.load_state_dict_from_url(
        #     "https://api.ngc.nvidia.com/v2/models/nvidia/waveglowpyt_fp32/versions/1/files/nvidia_waveglowpyt_fp32_20190306.pth",  # noqa: E501
        #     progress=True,
        #     # map_location=device,
        # )
        # state_dict = {key.replace("module.", ""): value for key, value in checkpoint["state_dict"].items()}

        # self.waveglow.load_state_dict(state_dict)
        # self.waveglow = waveglow.remove_weightnorm(waveglow)
        # self.waveglow.eval()

        # loss function
        self.criterion = torch.nn.MSELoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_loss = MeanSquaredError()
        self.val_loss = MeanSquaredError()
        self.test_loss = MeanSquaredError()

    def forward(self, x: torch.Tensor, wav_output=False):
        wav, y = None, self.net(x)
        # if wav_output:
        #     wav = self.waveglow(y)
        return self.net(x), wav

    def step(self, batch: Any):
        x, y, l = batch
        y_hat, _ = self.forward(x)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y, l

    def training_step(self, train_batch, batch_idx):
        loss, y_hat, y, l = self.step(train_batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # TODO: SignalToNoiseRatio or other audio metrics once we have incorporated a neural
        # vocoder

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": y_hat, "targets": y, "labels": l}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, val_batch, batch_idx):
        loss, y_hat, y, l = self.step(val_batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"loss": loss, "preds": y_hat, "targets": y, "labels": l}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        # TODO: SNR here
        loss, preds, targets, labels = self.step(batch)
        # log test metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return {"loss": loss, "preds": preds, "targets": targets, "labels": labels}

    def test_epoch_end(self, outputs: List[Any]):
        letter_res = {}
        letter_samples = {}
        for batch in outputs:
            for i in range(batch['labels'].shape[0]):
                letter = batch['labels'][i].item()
                if letter not in letter_res:
                    letter_res[letter] = torch.tensor(.0)
                    letter_samples[letter] = 0
                letter_res[letter] += batch['loss'].item()
                letter_samples[letter] += 1

        for letter, loss in letter_res.items():
            loss /= letter_samples[letter]
            letter = chr(ord('A')+letter)
            print(f"[*] Loss for '{letter}': MSE={loss:06f}")

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
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
