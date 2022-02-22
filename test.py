#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Testing script.
"""

from dataset import load_sign_alphabet
from models.lstm import Sign2SpeechNet
from models.autoencoder import VAE

import pytorch_lightning as pl
import torchaudio
import torch


def test_vae(model_path: str, input_path: str):
    '''
    Simple test (while waiting for a test set) to run inference on one pause.
    '''
    import numpy as np
    from torchvision.io import read_image, ImageReadMode, write_jpeg

    n_mels, n_fft, sample_rate = 64, 1024, 44100
    vae = VAE()
    model = vae.load_from_checkpoint(model_path)
    model.eval()
    sample_pose = torch.tensor(np.load(input_path).astype(np.float32)).unsqueeze(0)
    specto = model(sample_pose).detach()[0] # Spectogram
    print("Spectogram: ", specto.shape, specto.dtype)

    mel_inverter = torchaudio.transforms.InverseMelScale(sample_rate=sample_rate,
            n_stft=(n_fft//2)+1, n_mels=n_mels)
    griffin_limer = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=None)
    stft = mel_inverter(specto)
    print("STFT spectogram: ", stft.shape)
    pred_audio = griffin_limer(stft)
    write_jpeg(specto.type(dtype=torch.uint8), "test_specto.jpg")
    torchaudio.save("test.wav", pred_audio, sample_rate)


if __name__ == "__main__":
    # TODO: Use Hydra for experiment management and Weights&Biases for experiment logging
    test_vae("ae_ckpt/CZtyXaDbgTv3cfsXp42cSw/checkpoints/epoch=11-step=96599.ckpt", "dataset/train_poses/T/T  CY24.55P-0.08.npy")
