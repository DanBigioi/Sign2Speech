# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#
# Distributed under terms of the MIT license.


import numpy as np
import torch
import os

from torch.utils.data import Dataset, IterableDataset, DataLoader
from utils import read_poses_json
from typing import Tuple, Union


class SignAlphabetDataset(Dataset):
    def __init__(self, poses, labels, speech=None):
        self.poses = poses
        self.labels = labels
        self.spectograms = speech

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index) -> Tuple:
        if self.speech is not None:
            mfcc = self.speech[index]
            hand_poses = self.poses[index]
            label = self.labels[index]

            return mfcc, hand_poses, label

        else:
            hand_poses = self.poses[index]
            label = self.labels[index]
            return hand_poses, label


def load_sign_alphabet(
    alphabet_dataset_path, speech_gt_path, test=False
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    Load a Sign Alphabet dataset, split into training and validation or test sets, and return data
    loaders.
    If testing, a different dataset file is assumed and no splitting will be done.
    """
    poses, labels = read_poses_json(alphabet_dataset_path)
    print(f"[*] Loaded {poses.shape[0]} pose samples.")
    if not test:
        train_pct = 0.7
        print("[*] Splitting 70/30...")
        samples = dict()
        for pose, label in zip(poses, labels):
            if label not in samples:
                samples[label] = pose
            else:
                samples[label].append(pose)

        # TODO: Just change the dataset constructor so it just accepts a dict...
        # Split the samples with their labels into train and val
        train_poses, train_labels, val_poses, val_labels = [], [], [], []
        for label in samples.keys():
            count = train_pct*len(samples[label])
            train_poses.append(samples[label][:count])
            train_labels.append([label]*count)
            val_poses.append(samples[label][count:])
            val_labels.append([label]*len(samples[label]-count))

        train_loader = torch.utils.data.DataLoader(
            SignAlphabetDataset(train_poses, train_labels, speech_gt),
            num_workers=0,
            batch_size=32,
            shuffle=True,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            SignAlphabetDataset(val_poses, val_labels, speech_gt),
            num_workers=0,
            batch_size=32,
            shuffle=False,
            pin_memory=True,
        )
        return train_loader, val_loader
        return None, None
    else:
        return DataLoader(
            SignAlphabetDataset(poses, labels, speech_gt),
            num_workers=0,
            batch_size=32,
            shuffle=False,
            pin_memory=True,
        )

