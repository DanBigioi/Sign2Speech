#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Sign dataset of MFCCs for the LSTM.
"""


import torch

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ConvertImageDtype
from pytorch_lightning import LightningDataModule
from torchvision.transforms import transforms
from typing import Optional, Tuple

from src.datamodules.components.sign_component import (
    load_sign_alphabet,
    SignAlphabetMFCCDataset,
    ISLDataset,
)


class SignMFCCDataModule(LightningDataModule):
    def __init__(
        self,
        poses_dir: str,
        specto_dir: str,
        test_poses_dir: str,
        train_pct: tuple = 0.8,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        poses_src_fps: int = 30,
        poses_target_fps: int = 101,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        dataset = load_sign_alphabet(
            self.hparams.poses_dir,
            self.hparams.specto_dir,
            dataset_class=SignAlphabetMFCCDataset,
            poses_src_fps=self.hparams.poses_src_fps,
            poses_target_fps=self.hparams.poses_target_fps,
        )
        # TODO: When we get a test set we can load it separately in the same function
        # testset = load_sign_alphabet(
        #     self.hparams.data_dir, train=False, transform=self.transforms
        # )
        # dataset = ConcatDataset(datasets=[trainset, testset])
        train_length = int(self.hparams.train_pct * len(dataset))
        val_length = len(dataset) - train_length
        lengths = [
            train_length,
            val_length,
        ]
        if stage == "TrainerFn.FITTING" and not self.data_train and not self.data_val:
            self.data_train, self.data_val = random_split(
                dataset=dataset,
                lengths=lengths,
                generator=torch.Generator().manual_seed(42),
            )
        elif stage == "TrainerFn.TESTING" and not self.data_test:
            self.data_train, self.data_val = random_split(
                dataset=dataset,
                lengths=lengths,
                generator=torch.Generator().manual_seed(42),
            )
            self.data_test = self.data_val
            # load_sign_alphabet(
            #     self.hparams.test_poses_dir,
            #     self.hparams.specto_dir,
            #     dataset_class=ISLDataset,
            # )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            # batch_size=1,  # For our per-sign MSE
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
