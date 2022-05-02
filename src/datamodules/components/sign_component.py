#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


import numpy as np
import torch
import os

from torch.utils.data import Dataset
from torchvision.transforms import ConvertImageDtype
from torchvision.io import read_image, ImageReadMode
from sklearn import preprocessing
from typing import Tuple, Dict


class SignAlphabetSpectogramDataset(Dataset):
    def __init__(
        self, poses, labels, spectograms, transforms=ConvertImageDtype(torch.float32)
    ):
        self.poses = poses
        self.labels = labels
        self.spectograms = spectograms
        self.transforms = transforms

    def __len__(self):
        assert len(self.poses) == len(self.labels) and len(self.poses) == len(
            self.spectograms
        ), "Dataset items mismatch"
        return len(self.poses)

    def __getitem__(self, idx) -> Tuple:
        # TODO: Memory pinning?
        specto = read_image(self.spectograms[idx], mode=ImageReadMode.GRAY)
        if self.transforms:
            specto = self.transforms(specto)
        hand_pose = np.load(self.poses[idx]).astype(np.float32)
        label = self.labels[idx]
        return hand_pose, specto, label


class SignAlphabetMFCCDataset(Dataset):
    def __init__(self, poses, labels, mfccs):
        self.poses = poses
        self.labels = labels
        self.mfccs = mfccs

    def __len__(self):
        assert len(self.poses) == len(self.labels) and len(self.poses) == len(
            self.mfccs
        ), "Dataset items mismatch"
        return len(self.poses)

    def __getitem__(self, idx) -> Tuple:
        # TODO: Memory pinning?
        hand_pose = np.load(self.poses[idx])
        mfccs = self.mfccs[idx]
        label = self.labels[idx]
        return mfccs, hand_pose, label


def parse_numpy_dataset(root: str, verbose=False) -> Dict[str, str]:
    """
    Parses a dataset root directory and returns a dictionary with key "alphabet label" and value
    "list of hand poses".
    """
    samples = dict()
    n_samples = 0
    for dirname in os.listdir(root):
        if not os.path.isdir(os.path.join(root, dirname)):
            continue
        assert dirname not in samples, f"{dirname} already in dict"
        samples[dirname] = []
        for root_dir, _, files in os.walk(os.path.join(root, dirname)):
            for file_name in files:
                file_path = os.path.join(root_dir, file_name)
                if os.path.isfile(file_path) and file_path.endswith(".npy"):
                    samples[dirname].append(file_path)
        assert len(samples[dirname]) > 0, f"No samples found for label '{dirname}'"
        if verbose:
            print(f"-> Loaded {len(samples[dirname])} samples for label '{dirname}'")
        n_samples += len(samples[dirname])
    assert len(list(samples.keys())) > 0, f"Empty dataset"
    print(f"[*] Loaded {n_samples} pose samples.")
    return samples


def load_sign_alphabet(
    alphabet_dataset_path, speech_gt_path, transforms, train=True
) -> Dataset:
    """
    Load a Sign Alphabet dataset, split into training and validation or test sets, and return data
    loaders.
    If testing, a different dataset file is assumed and no splitting will be done.
    """
    samples = parse_numpy_dataset(alphabet_dataset_path)
    le = preprocessing.LabelEncoder()
    label_codes = le.fit_transform(np.array(list(samples.keys())))

    speech_gt = dict()
    for path in os.listdir(speech_gt_path):
        full_path = os.path.join(speech_gt_path, path)
        if os.path.isfile(full_path):
            speech_gt[path.split(".")[0].upper()] = full_path
    assert len(speech_gt) > 0, "Speech ground truth not loaded"

    poses, labels, spectograms = [], [], []
    for i, label in enumerate(samples.keys()):
        poses += samples[label]
        labels += [label_codes[i]] * len(samples[label])
        spectograms += [speech_gt[label.upper()]] * len(samples[label])

    return SignAlphabetSpectogramDataset(poses, labels, spectograms, transforms=transforms)
