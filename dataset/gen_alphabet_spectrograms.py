#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Generate one spectrogram per letter from a dataset of wav files.
"""

import torchaudio
import argparse
import random
import torch
import os

from torchvision.utils import save_image
from torchaudio import transforms
from tqdm import tqdm


def wav_to_spectrogram(wav) -> torch.Tensor:
    def pad_trunc(aud, max_ms):
        """
        Taken from https://ketanhdoshi.github.io/Audio-Classification/
        """
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms

        if sig_len > max_len:
            # Truncate the signal to the given length
            sig = sig[:, :max_len]

        elif sig_len < max_len:
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)
        return sig, sr

    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        """
        Taken from https://ketanhdoshi.github.io/Audio-Classification/
        """
        sig, sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(
            sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels
        )(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return spec

    audio = torchaudio.load(wav)
    audio = pad_trunc(audio, 1000)
    return spectro_gram(audio)


def generate(wav_dir, spec_dir):
    if not os.path.exists(spec_dir):
        os.makedirs(spec_dir)
    for file in tqdm(os.listdir(wav_dir)):
        full_path = os.path.join(wav_dir, file)
        if not os.path.isfile(full_path):
            continue
        spec = wav_to_spectrogram(full_path)
        print(spec.shape)
        save_image(spec, os.path.join(spec_dir, f"{file.split('.')[0]}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Folder of WAV files", type=str)
    parser.add_argument(
        "dest", help="Destination folder of spectrogram images", type=str
    )
    args = parser.parse_args()
    print("[*] Generating mono sound Mel spectrograms...")
    generate(args.source, args.dest)

