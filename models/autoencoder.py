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

import torch.nn.functional as F
import pytorch_lightning as pl
import torch


class VAE(pl.LightningModule):
    # TODO: The variational part of VAE
    def __init__(self, input_dim: int = 63, latent_dim: int = 16):
        super().__init__()
        assert latent_dim < 32
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, latent_dim),
            nn.ReLU(True),
            # nn.Dropout2d(p=0.2),
        )
        self.projector = nn.Sequential(nn.Linear(latent_dim, 32), nn.ReLU(True), nn.Linear(32, 64),
                nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=(5, 6), stride=2), # C=8,H=11,W=19
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, kernel_size=(3, 5), stride=2, output_padding=(0,1)),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 2, kernel_size=(3, 4), stride=2, output_padding=(0,0)),
            nn.ReLU(),
            nn.ConvTranspose2d(2, 1, kernel_size=(3, 3), stride=2, output_padding=(1,1)),
            nn.ReLU(),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        return self.decoder(embedding)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y, l = train_batch
        z = self.encoder(x)
        z = self.projector(z)
        z = z.view(z.size(0), 16, 2, -1) # Reshape to (bs, 16, 2, 2)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y, l = val_batch
        z = self.encoder(x)
        z = self.projector(z)
        z = z.view(z.size(0), 16, 2, -1) # Reshape to (bs, 16, 2, 2)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, y)
        self.log("val_loss", loss)
        return loss

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
