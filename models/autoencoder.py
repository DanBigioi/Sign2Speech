#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Autoencoder models.
"""


from typing import Tuple
from torch import nn

import pytorch_lightning as pl
import torch.functional as F
import torch


class VAE(pl.LightningModule):
    def __init__(self, input_dim: int = 63, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.Linear(64, latent_dim))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(),
            nn.ConvTranspose2d(6,3,kernel_size=5),
            nn.ReLU())

    def forward(self, x):
        embedding = self.encoder(x)
        return self.decoder(embedding)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        print("TRAINING STEP")
        x, y = train_batch
        print(x, y)
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        print(type(val_batch))
        x, y = val_batch
        x = x .view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)
        return loss

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        loss.backward()
