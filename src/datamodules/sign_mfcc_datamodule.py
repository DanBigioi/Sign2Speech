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
)


class SignMFCCDataModule(LightningDataModule):
    def __init__(
        self,
        poses_dir: str,
        specto_dir: str,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose([ConvertImageDtype(torch.float32)])

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
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = load_sign_alphabet(
                self.hparams.poses_dir,
                self.hparams.specto_dir,
                dataset_class=SignAlphabetMFCCDataset,
                transforms=self.transforms,
            )
            # TODO: When we get a test set we can load it separately in the same function
            # testset = load_sign_alphabet(
            #     self.hparams.data_dir, train=False, transform=self.transforms
            # )
            # dataset = ConcatDataset(datasets=[trainset, testset])
            dataset = trainset
            train_length = int(self.hparams.train_val_test_split[0] * len(dataset))
            val_length = int(self.hparams.train_val_test_split[1] * len(dataset))
            lengths = [
                train_length,
                val_length,
                len(dataset) - (train_length + val_length),
            ]
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=lengths,
                generator=torch.Generator().manual_seed(42),
            )

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
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
