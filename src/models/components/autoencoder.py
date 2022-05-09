#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

from torch import nn


class AE_Deconv(nn.Module):
    def __init__(self, input_dim: int = 63, latent_dim: int = 16):
        super().__init__()
        assert latent_dim < 32
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(True),
            # nn.Dropout2d(p=0.2),
        )
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(True), nn.Linear(32, 64), nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=(5, 6), stride=2),  # C=8,H=11,W=19
            nn.ReLU(),
            nn.ConvTranspose2d(
                8, 4, kernel_size=(3, 5), stride=2, output_padding=(0, 1)
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                4, 2, kernel_size=(3, 4), stride=2, output_padding=(0, 0)
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                2, 1, kernel_size=(3, 3), stride=2, output_padding=(1, 1)
            ),
        )

    def forward(self, x):
        embedding = self.encoder(x)
        z = self.projector(embedding)
        z = z.view(z.size(0), 16, 2, -1)  # Reshape to (bs, 16, 2, 2)
        return self.decoder(z)


class AE(nn.Module):
    def __init__(self, input_dim: int = 63, latent_dim: int = 16):
        super().__init__()
        assert latent_dim < 32
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(True),
            nn.Linear(32, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(True),
            # nn.Dropout2d(p=0.2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 5504),
        )

    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.decoder(z)
        return y_hat.view(y_hat.size(0), 1, 64, 86)
