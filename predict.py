#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Script to run single inference on a given input.
"""

from src.models.autoencoder_module import AELitModule

import numpy as np
import torchaudio
import torch
import sys


def predict(input: str):
    # ckpt can be also a URL!
    CKPT_PATH = "models/last.ckpt"
    n_fft, n_mels, s_rate = 1024, 80, 44100

    # load model from checkpoint
    # model __init__ parameters will be loaded from ckpt automatically
    # you can also pass some parameter explicitly to override it
    trained_model = AELitModule.load_from_checkpoint(checkpoint_path=CKPT_PATH)

    # switch to evaluation mode
    trained_model.eval()
    trained_model.freeze()

    # Load the input
    x = torch.from_numpy(np.load(input).astype(np.float32))
    spectogram, waveform = trained_model(x.unsqueeze(0), wav_output=True)
    audio, sample_rate = waveform
    torchaudio.save("test_voco.wav", audio.cpu(), sample_rate)

    mel_inverter = torchaudio.transforms.InverseMelScale(sample_rate=s_rate,
            n_stft=(n_fft//2)+1, n_mels=n_mels)
    griffin_limer = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=None)
    stft = mel_inverter(spectogram)
    print("STFT spectogram: ", stft.shape)
    pred_audio = griffin_limer(stft)
    print(pred_audio.shape)
    torchaudio.save("test_specto.wav", pred_audio, s_rate)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input>")
        exit(1)
    predict(sys.argv[1])
