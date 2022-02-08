#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Training script.
"""

from dataset import load_sign_alphabet
from models.autoencoder import VAE

import pytorch_lightning as pl


def train():
    train_loader, val_loader = load_sign_alphabet("dataset/train_poses/", "dataset/spec/")
    model = VAE()
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    # TODO: Use Hydra for experiment management and Weights&Biases for experiment logging
    train()
