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

from torchvision.transforms import ConvertImageDtype
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset
from typing import Tuple, Dict, List
from sklearn import preprocessing

from src.vendor.dataio import AudioFile, ImplicitAudioWrapper


class SignAlphabetWaveformDataset(Dataset):
    def __init__(self, poses, labels, waveforms):
        self.poses = poses
        self.labels = labels
        self.audio_datasets = self._load_audio_datasets(waveforms)

    def _load_audio_datasets(self, waveforms: List):
        datasets = []
        print(f"Loading {len(waveforms)} waveforms")
        for wav in waveforms:
            audio_ds = AudioFile(filename=wav)
            datasets.append(ImplicitAudioWrapper(audio_ds))
        return datasets

    def __len__(self):
        assert len(self.poses) == len(self.labels), "Dataset items mismatch"
        return len(self.poses)

    def __getitem__(self, idx) -> Tuple:
        """
        Return values:
        hand_pose: numpy array representing a hand pose.
        wav_dataset: returns a Torch Dataset that allows to iterate over a waveform and obtain such
            dictionaries tuple:
            {
                idx: the querried index,
                coords: a linearly spaced 1D grid from -100 to 100 with the number of WAV sumples,
            },
            {
                func: the WAV data (amplitudes),
                rate: the WAV sample rate,
                scale: the scale of the amplitudes (max of abs value of the data),
            }
        label: the alphabet letter (int)
        """
        # TODO: Memory pinning?
        idx = 0  # Debug
        hand_pose = np.load(self.poses[idx]).astype(np.float32)
        label = self.labels[idx]
        # Indexing has no effect, it will return the whole audio range
        wav_dataset = self.audio_datasets[label][0]
        return (
            {"pose": hand_pose, "audio": wav_dataset[0]["coords"]},
            {"amplitude": wav_dataset[1]["func"]},
            label,
        )


class SignAlphabetSpectogramDataset(Dataset):
    def __init__(
        self, poses, labels, spectograms, transforms=ConvertImageDtype(torch.float32)
    ):
        self.poses = poses
        self.labels = labels
        self.spectograms = spectograms
        self.transforms = transforms

    def __len__(self):
        assert len(self.poses) == len(self.labels), "Dataset items mismatch"
        return len(self.poses)

    def __getitem__(self, idx) -> Tuple:
        # TODO: Memory pinning?
        hand_pose = np.load(self.poses[idx]).astype(np.float32)
        label = self.labels[idx]
        specto = read_image(self.spectograms[label], mode=ImageReadMode.GRAY)
        if self.transforms:
            specto = self.transforms(specto)
        return hand_pose, specto, label


class SignAlphabetMFCCDataset(Dataset):
    def __init__(self, poses, labels, mfccs, src_fps, target_fps):
        self.poses = []
        for pose in poses:
            self.poses.append(interp_func(
                np.load(pose).astype(np.float32), src_fps=src_fps, trg_fps=target_fps
            ).astype(np.float32))
        self.labels = labels
        self.mfccs = mfccs

    def __len__(self):
        assert len(self.poses) == len(self.labels), "Dataset items mismatch"
        return len(self.poses)

    def __getitem__(self, idx) -> Tuple:
        hand_pose = self.poses[idx]
        label = self.labels[idx]
        mel_coef = np.load(self.mfccs[label]).astype(np.float32).swapaxes(1, 2)
        return hand_pose, mel_coef, label


def interp_func(input_mat, src_fps=30, trg_fps=101):
    xp = list(np.arange(0, input_mat.shape[0], 1))
    interp_xp = list(np.arange(0, input_mat.shape[0], src_fps / trg_fps))
    interp_mat = np.zeros(shape=(len(interp_xp), input_mat.shape[1]))
    for j in range(input_mat.shape[1]):
        interp_mat[:, j] = np.interp(interp_xp, xp, input_mat[:, j])
    return interp_mat


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
    alphabet_dataset_path,
    gt_path,
    dataset_class: Dataset,
    transforms=None,
    train=True,
    poses_src_fps=None,
    poses_target_fps=None,
) -> Dataset:
    """
    Load a Sign Alphabet dataset, split into training and validation or test sets, and return data
    loaders.
    If testing, a different dataset file is assumed and no splitting will be done.
    """
    # TODO: Load the test set
    samples = parse_numpy_dataset(alphabet_dataset_path)
    le = preprocessing.LabelEncoder()
    label_codes = le.fit_transform(np.array(list(samples.keys())))

    speech_gt = dict()
    for path in os.listdir(gt_path):
        full_path = os.path.join(gt_path, path)
        if os.path.isfile(full_path):
            speech_gt[path.split(".")[0].upper()] = full_path
    assert len(speech_gt) > 0, "Speech ground truth not loaded"

    poses, labels, gt = [], [], []
    for i, label in enumerate(samples.keys()):
        poses += samples[label]
        labels += [label_codes[i]] * len(samples[label])
        gt += [speech_gt[label.upper()]]

    dataset = None
    if dataset_class is SignAlphabetSpectogramDataset:
        dataset = SignAlphabetSpectogramDataset(
            poses, labels, gt, transforms=transforms
        )
    elif dataset_class is SignAlphabetWaveformDataset:
        dataset = SignAlphabetWaveformDataset(poses, labels, gt)
    elif dataset_class is SignAlphabetMFCCDataset:
        dataset = SignAlphabetMFCCDataset(
            poses, labels, gt, poses_src_fps, poses_target_fps
        )
    return dataset
