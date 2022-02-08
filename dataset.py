# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
#
# Distributed under terms of the MIT license.


import numpy as np
import torch
import os

from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision.io import read_image
from typing import Tuple, Union, Dict
from sklearn import preprocessing
from utils import read_poses_json


class SignAlphabetDataset(Dataset):
    def __init__(self, poses, labels, spectograms=None):
        self.poses = poses
        self.labels = labels
        self.spectograms = spectograms

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index) -> Tuple:
        # TODO: Move things to CUDA?
        # TODO: Image to tensor (load the image with torchvision?)
        if self.spectograms is not None:
            specto = read_image(self.spectograms[index])
            hand_poses = self.poses[index]
            label = self.labels[index]

            return specto, hand_poses, label

        else:
            hand_poses = self.poses[index]
            label = self.labels[index]
            return hand_poses, label


def parse_dataset(root: str, verbose = False) -> Dict[str, str]:
    """
    Parses a dataset root directory and returns a dictionary with key "alphabet label" and value
    "list of hand poses".
    """
    samples = dict()
    n_samples = 0
    for dirname in os.listdir(root):
        if not os.path.isdir(os.path.join(root, dirname)):
            continue
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
    alphabet_dataset_path, speech_gt_path, test=False
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """
    Load a Sign Alphabet dataset, split into training and validation or test sets, and return data
    loaders.
    If testing, a different dataset file is assumed and no splitting will be done.
    """
    samples = parse_dataset(alphabet_dataset_path)
    le = preprocessing.LabelEncoder()
    label_codes = le.fit_transform(np.array(list(samples.keys())))

    speech_gt = []
    for path in os.listdir(speech_gt_path):
        if os.path.isfile(path):
            speech_gt.append(path)

    if not test:
        train_pct = 0.7
        print("[*] Splitting 70/30...")

        # Split the samples with their labels into train and val
        train_poses, train_labels, val_poses, val_labels = [], [], [], []
        for i, label in enumerate(samples.keys()):
            count = int(train_pct*len(samples[label]))
            train_poses.append(samples[label][:count])
            train_labels.append([label_codes[i]]*count)
            val_poses.append(samples[label][count:])
            val_labels.append([label_codes[i]]*(len(samples[label])-count))

        train_loader = DataLoader(
            SignAlphabetDataset(train_poses, train_labels, speech_gt),
            num_workers=0,
            batch_size=32,
            shuffle=True,
            pin_memory=True,
        )
        val_loader = DataLoader(
            SignAlphabetDataset(val_poses, val_labels, speech_gt),
            num_workers=0,
            batch_size=32,
            shuffle=False,
            pin_memory=True,
        )
        return train_loader, val_loader
    else:
        poses, labels = [], []
        for i, label in enumerate(samples.keys()):
            poses.append(samples[label])
            labels.append([label_codes[i]]*len(samples[label]))

        return DataLoader(
            SignAlphabetDataset(poses, labels, speech_gt),
            num_workers=0,
            batch_size=32,
            shuffle=False,
            pin_memory=True,
        )

