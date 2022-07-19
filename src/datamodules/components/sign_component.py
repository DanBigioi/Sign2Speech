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
            self.poses.append(
                interp_func(
                    np.load(pose).astype(np.float32),
                    src_fps=src_fps,
                    trg_fps=target_fps,
                ).astype(np.float32)
            )
        self.labels = labels
        self.mfccs = mfccs

    def __len__(self):
        assert len(self.poses) == len(self.labels), "Dataset items mismatch"
        return len(self.poses)

    def __getitem__(self, idx) -> Tuple:
        hand_pose = self.poses[idx]
        label = self.labels[idx]
        mel_coef = np.load(self.mfccs[label]).astype(np.float32)  # .swapaxes(1, 2)
        return hand_pose, mel_coef, label


def interp_func(input_mat, src_fps=30, trg_fps=101):
    if src_fps == trg_fps:
        return input_mat
    xp = list(np.arange(0, input_mat.shape[0], 1))
    interp_xp = list(np.arange(0, input_mat.shape[0], src_fps / trg_fps))
    interp_mat = np.zeros(shape=(len(interp_xp), input_mat.shape[1]))
    for j in range(input_mat.shape[1]):
        interp_mat[:, j] = np.interp(interp_xp, xp, input_mat[:, j])
    return interp_mat


class ISLDataset(Dataset):
    def __init__(self, poses, labels, mfccs):
        self.poses = []
        for pose in poses:
            self.poses.append(self._preprocess(np.load(pose).astype(np.float32)))
        self.labels = labels
        self.mfccs = mfccs

    def _preprocess(self, pose_seq: np.ndarray) -> np.ndarray:
        def putPoseInUnitCube(pose):
            # Find the min and max X and Y coord of all the poses. This is the bounding box within which all pose hands will fit.
            # (Use np slicing. [;, iterates through all rows, 0::3] takes every 3rd value from first col for Xs, then 1::3 for Ys)
            minX = np.min(pose[0::3])
            minY = np.min(pose[1::3])
            minZ = np.min(pose[2::3])
            maxX = np.max(pose[0::3])
            maxY = np.max(pose[1::3])
            maxZ = np.max(pose[2::3])

            # The bounding box is the maxmin differences. This is in Blender coord units (metres).
            boundingBoxW = abs(
                maxX - minX
            )  # width range  !!!  X IS ACTUALLY THE HEIGHT IN BLENDER !!!!
            boundingBoxH = abs(maxY - minY)  # height range
            boundingBoxD = abs(maxZ - minZ)  # depth range

            # the bounding cube side is the largest of the three (hte cube is equal-sided)
            boundingCubeSide = np.max([boundingBoxW, boundingBoxH, boundingBoxD])
            # print(boundingBoxH, boundingBoxW, boundingBoxD, boundingCubeSide)

            # Move the pose so that all the points are positive - so in +ive space
            pose[0::3] = pose[0::3] - minX
            pose[1::3] = pose[1::3] - minY
            pose[2::3] = pose[2::3] - minZ

            # Scale the cube to be the unit size eg 1x1x1
            cube = (1, 1, 1)
            pose[0::3] = (pose[0::3] / boundingCubeSide) * cube[0]  # Scale X coords
            pose[1::3] = (pose[1::3] / boundingCubeSide) * cube[1]  # Scale Y coords
            pose[2::3] = (pose[2::3] / boundingCubeSide) * cube[2]  # Scale z coords

            return pose

        # Seems to be x=width, y=height, z=depth!
        # Wrist alignment
        new_pose_seq = []
        for i in range(pose_seq.shape[0]):
            pose = pose_seq[i].reshape(21, 3)
            # Wrist alginment:
            aligned_pose = pose - pose[0]
            aligned_pose = -aligned_pose # Mirror the hand
            new_pose_seq.append(aligned_pose)
        pose_seq = np.array(new_pose_seq)
        # Positive axis enforcement
        new_pose_seq = []
        for i in range(pose_seq.shape[0]):
            # pose = pose_seq[i].reshape(21, 3)
            pose = putPoseInUnitCube(pose_seq[i].flatten())
            new_pose_seq.append(pose)
        pose_seq = np.array(new_pose_seq)
        return pose_seq

    def __len__(self):
        assert len(self.poses) == len(self.labels), "Dataset items mismatch"
        return len(self.poses)

    def __getitem__(self, idx) -> Tuple:
        hand_pose = self.poses[idx]
        label = self.labels[idx]
        mel_coef = np.load(self.mfccs[label]).astype(np.float32)  # .swapaxes(1, 2)
        return hand_pose, mel_coef, label


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
    elif dataset_class is ISLDataset:
        dataset = ISLDataset(poses, labels, gt)
    return dataset
