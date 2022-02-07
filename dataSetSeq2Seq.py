import torch
import numpy as np
import os
from torch.utils.data import Dataset, IterableDataset, DataLoader
from utils import readKeyPointJson

class dataSetSeq2Seq(Dataset):
    def __init__(self, keypoints,  labels, speech = None):
        self.keypoints = keypoints
        self.labels = labels
        self.speech = speech

    def __len__(self):
        return len(self.keypoints)

    def __getitem__(self, index):
        if self.speech is not None:
            mfcc = self.speech[index]
            hand_keypoints = self.keypoints[index]
            label = self.labels[index]

            return mfcc, hand_keypoints, label

        else:
            hand_keypoints = self.keypoints[index]
            label = self.labels[index]
            return hand_keypoints, label


def prepare_dataloader(keypoints,  labels, speech):
    data_loader = torch.utils.data.DataLoader(
        dataSetSeq2Seq(keypoints,  labels, speech),
        num_workers=0,
        batch_size=10,
        shuffle=True,
        collate_fn=None,
        pin_memory=True,
    )

    return data_loader


##################Dummy Training Loop########################################
'''
keypoint_array, label_array = readKeyPointJson('DataSet.json')
print(keypoint_array.shape)
print(label_array.shape)
dataloader = prepare_dataloader(keypoint_array, label_array, speech=None)
NUM_EPOCHS = 20

for i in range(NUM_EPOCHS):
    print(i)
    for i,data in enumerate(dataloader):
        hand_keypoints, labels = data
        print("shape of keypoints = " + str(hand_keypoints.shape))
        print("shape of labels = " + str(labels.shape))
        print(i)
'''
##############################################################################