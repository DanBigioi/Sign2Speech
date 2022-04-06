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

from torchvision.io import read_image, ImageReadMode, write_jpeg

from dataset import load_sign_alphabet
from models.lstm import Sign2SpeechNet
from models.autoencoder import VAE, AE

import pytorch_lightning as pl
import numpy as np
import torchaudio
import torch

def test_gt(input_path: str):
    '''
    Simple test (while waiting for a test set) to run inference on one pose.
    '''
    n_mels, n_fft, sample_rate = 64, 1024, 44100
    specto = read_image(input_path, mode=ImageReadMode.GRAY).type(dtype=torch.float32)
    print("Spectogram: ", specto.shape, specto.dtype)

    mel_inverter = torchaudio.transforms.InverseMelScale(sample_rate=sample_rate,
            n_stft=(n_fft//2)+1, n_mels=n_mels)
    griffin_limer = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=None)
    stft = mel_inverter(specto)
    print("STFT spectogram: ", stft.shape)
    pred_audio = griffin_limer(stft)
    torchaudio.save("gt.wav", pred_audio, sample_rate)

def test_vae(model_path: str, input_path: str):
    '''
    Simple test (while waiting for a test set) to run inference on one pose.
    '''
    n_mels, n_fft, sample_rate = 64, 1024, 44100
    vae = AE()
    model = vae.load_from_checkpoint(model_path)
    model.eval()
    sample_pose = torch.tensor(np.load(input_path).astype(np.float32)).unsqueeze(0)
    specto = model(sample_pose).detach()[0] # Spectogram
    print("Spectogram: ", specto.shape, specto.dtype)
    write_jpeg(specto.type(dtype=torch.uint8), "test_specto.png")

    mel_inverter = torchaudio.transforms.InverseMelScale(sample_rate=sample_rate,
            n_stft=(n_fft//2)+1, n_mels=n_mels)
    griffin_limer = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=None)
    stft = mel_inverter(specto)
    print("STFT spectogram: ", stft.shape)
    pred_audio = griffin_limer(stft)
    torchaudio.save("test.wav", pred_audio, sample_rate)


if __name__ == "__main__":
    # TODO: Use Hydra for experiment management and Weights&Biases for experiment logging
    # test_gt("dataset/spec/h.png")
    test_vae("ae_ckpt/jbSfjzSfqKKHpRv2mkNsDh/checkpoints/epoch=6-step=1763.ckpt", "dataset/train_poses/T/T  CY24.55P-0.08.npy")
