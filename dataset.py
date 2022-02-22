# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#
# Distributed under terms of the MIT license.


import numpy as np
import torch
import os

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import ToTensor, ConvertImageDtype
from torchvision.io import read_image, ImageReadMode
from typing import Tuple, Union, Dict
from sklearn import preprocessing


class SignAlphabetSpectogramDataset(Dataset):
    def __init__(
        self, poses, labels, spectograms, transform=ConvertImageDtype(torch.float32)
    ):
        self.poses = poses
        self.labels = labels
        self.spectograms = spectograms
        self.transform = transform

    def __len__(self):
        assert len(self.poses) == len(self.labels) and len(self.poses) == len(
            self.spectograms
        ), "Dataset items mismatch"
        return len(self.poses)

    def __getitem__(self, idx) -> Tuple:
        # TODO: Memory pinning?
        specto = read_image(self.spectograms[idx], mode=ImageReadMode.GRAY)
        if self.transform:
            specto = self.transform(specto)
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
    alphabet_dataset_path, speech_gt_path, batch_size=32, train_pct=0.7, test=False
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
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
    dataset = SignAlphabetSpectogramDataset(poses, labels, spectograms)

    if not test:
        print("[*] Splitting 70/30...")
        train_sz = int(0.7 * len(dataset))
        train, val = random_split(dataset, [train_sz, len(dataset) - train_sz])
        print(f"-> {len(train)} training samples.")
        print(f"-> {len(val)} validation samples.")

        train_loader = DataLoader(
            train,
            num_workers=8,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val,
            num_workers=8,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
        return train_loader, val_loader
    else:
        poses, labels = [], []
        for i, label in enumerate(samples.keys()):
            poses.append(samples[label])
            labels.append([label_codes[i]] * len(samples[label]))

        return DataLoader(
            dataset,
            num_workers=8,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
        )
