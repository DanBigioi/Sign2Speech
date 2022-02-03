# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#
# Distributed under terms of the MIT license.


import numpy as np
import torch
import os

from torch.utils.data import Dataset, IterableDataset, DataLoader
from utils import read_keypoints_json
from typing import Tuple, Union


class SignAlphabetDataset(Dataset):
    def __init__(self, keypoints, labels, speech=None):
        self.keypoints = keypoints
        self.labels = labels
        self.spectograms = speech

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, index) -> Tuple:
        if self.speech is not None:
            mfcc = self.speech[index]
            hand_keypoints = self.keypoints[index]
            label = self.labels[index]

            return mfcc, hand_keypoints, label

        else:
            hand_keypoints = self.keypoints[index]
            label = self.labels[index]
            return hand_keypoints, label


def load_sign_alphabet(
    alphabet_dataset_path, speech_gt_path, test=False
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    Load a Sign Alphabet dataset, split into training and validation or test sets, and return data
    loaders.
    If testing, a different dataset file is assumed and no splitting will be done.
    """
    keypoints, labels = read_keypoints_json(alphabet_dataset_path)
    print(f"[*] Loaded {keypoints.shape[0]} pose samples.")
    if not test:
        train_pct = 0.7
        print("[*] Splitting 70/30...")
        samples = dict()
        for pose, label in zip(keypoints, labels):
            if label not in samples:
                samples[label] = pose
            else:
                samples[label].append(pose)


        # train_loader = torch.utils.data.DataLoader(
        #     SignAlphabetDataset(train_keypoints, train_labels, speech_gt),
        #     num_workers=0,
        #     batch_size=32,
        #     shuffle=True,
        #     pin_memory=True,
        # )
        # val_loader = torch.utils.data.DataLoader(
        #     SignAlphabetDataset(val_keypoints, val_labels, speech_gt),
        #     num_workers=0,
        #     batch_size=32,
        #     shuffle=False,
        #     pin_memory=True,
        # )
        # return train_loader, val_loader
        return None, None
    else:
        return None
        # return DataLoader(
         #    SignAlphabetDataset(keypoints, labels, speech_gt),
         #    num_workers=0,
         #    batch_size=32,
         #    shuffle=False,
         #    pin_memory=True,
        # )

