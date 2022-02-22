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

from pytorch_lightning.loggers import WandbLogger

from dataset import load_sign_alphabet
from models.lstm import Sign2SpeechNet
from models.autoencoder import VAE

import pytorch_lightning as pl
import torchaudio
import torch


def train_vae(restore_from: str = None):
    wandb_logger = WandbLogger(project="VAE Sign2Speech", log_model="all")
    train_loader, val_loader = load_sign_alphabet(
        "dataset/train_poses/", "dataset/spec/", batch_size=2
    )
    vae = VAE()
    # Save checkpoints to './ae_ckpt/'
    trainer = pl.Trainer(gpus=1, default_root_dir="ae_ckpt/", logger=wandb_logger)
    wandb_logger.watch(vae)
    trainer.fit(vae, train_loader, val_loader, ckpt_path=restore_from)


def eval_vae(model_path: str):
    vae = VAE()
    model = vae.load_from_checkpoint(model_path)
    model.eval()

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


def train_audio2landmark(
    num_epochs=50000,
    learning_rate=0.001,
    load_model=False,
    input_size=89,
    hidden_size=128,
    num_layers=4,
    bidirectional=True,
):
    # Tensorboard
    trainDir = "C:/Users/ionut/Documents/My Python Projects/audio2video/Train_Directory"
    os.chdir(trainDir)
    writer = SummaryWriter("runs/loss_plot")
    model = Sign2SpeechNet(input_size, hidden_size, num_layers, bidirectional)
    train_mfcc_list = np.load(
        "C:/Users/ionut/Documents/BBC Lip Reading DataSet/TED DataSet Padded Audio and Landmarks/Padded_Audio/padded_audio_array.npy"
    )[:17000]
    train_landmark_list = np.load(
        "C:/Users/ionut/Documents/BBC Lip Reading DataSet/TED DataSet Padded Audio and Landmarks/Padded_Landmarks/padded_landmark_array.npy"
    )[:17000]

    train_dataloader = dataSetSeq2Seq.prepare_dataloader(
        train_mfcc_list, train_landmark_list, 1
    )

    validation_landmark_list = np.load(
        "C:/Users/ionut/Documents/BBC Lip Reading DataSet/TED DataSet Padded Audio and Landmarks/Padded_Landmarks/padded_landmark_array.npy"
    )[17000:]
    validation_mfcc_list = np.load(
        "C:/Users/ionut/Documents/BBC Lip Reading DataSet/TED DataSet Padded Audio and Landmarks/Padded_Audio/padded_audio_array.npy"
    )[17000:]
    validation_dataloader = dataSetSeq2Seq.prepare_dataloader(
        validation_mfcc_list, validation_landmark_list, 1
    )


if __name__ == "__main__":
    # TODO: Use Hydra for experiment management and Weights&Biases for experiment logging
    # train_vae()
    test_vae("ae_ckpt/CZtyXaDbgTv3cfsXp42cSw/checkpoints/epoch=11-step=96599.ckpt", "dataset/train_poses/T/T  CY24.55P-0.08.npy")
