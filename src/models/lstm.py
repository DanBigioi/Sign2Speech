# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
LSTM-based models.
"""

# TODO: Refactor this into a module + component!

from typing import Tuple
from torch import nn

import pytorch_lightning as pl
import torch.functional as F
import torch


class Sign2SpeechNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size  # for lstm
        self.num_layers = num_layers  # for lstm
        self.input_size = input_size  # Should be set to the number of keypoints
        self.bilstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.5,  # used to be 0.5
            bidirectional=bidirectional,
            batch_first=True,
        )

    def forward(self, mfcc, hand_keypoints, label):
        # If input is sequential
        # In Shape = Batch * Sequence_Len * Keypoint Shape
        output, (hn, cn) = self.bilstm(hand_keypoints)

        # if input is non sequential
        # In Shape = Batch * Keypoint Shape

        # TODO: Figure out what network architecture we want to use

        # Out Shape = Batch * Sequence_Len * MFCC Features
        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        mfcc, hand_keypoints, label = batch
        mfcc = mfcc.float()
        hand_keypoints = hand_keypoints.float()
        predicted_mfcc = self.forward(mfcc, hand_keypoints, label)
        loss = F.mse_loss(predicted_mfcc, mfcc)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        mfcc, hand_keypoints, label = batch
        mfcc = mfcc.float()
        hand_keypoints = hand_keypoints.float()
        predicted_mfcc = self.forward(mfcc, hand_keypoints, label)
        loss = F.mse_loss(predicted_mfcc, mfcc)
        self.log("val_loss", loss)
        return loss

    def backward(self, trainer, loss, optimizer, optimizer_idx):
        loss.backward()
