import json
import numpy as np
from sklearn import preprocessing
import torch

def readKeyPointJson(jsFile):

    f = open(jsFile)
    data = json.load(f)
    keypoint_array = []
    label_array = []
    for i in data['pose']:
        keypoint_array.append(i)

    for i in data['label']:
        label_array.append(i[1])

    #Encoding the labels so that pytorch is able to trasnform them into tensors
    label_array = np.asarray(label_array)
    le = preprocessing.LabelEncoder()
    label_array = le.fit_transform(label_array)

    keypoint_array = np.asarray(keypoint_array)
    keypoint_array = np.reshape(keypoint_array, (320, 63))

    f.close()

    return keypoint_array, label_array

def loadMFCC(filepath):
    mfcc = (np.load(filepath))
    print('mfcc shape = '+str(mfcc.shape))
    return mfcc

def loadLandmarks(filepath):
    landmarks = (np.load(filepath))
    print('landmarks shape = '+str(landmarks.shape))
    return landmarks

def save_checkpoint(state, filename="my_checkpointTest.pt"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])



keypoint, label = readKeyPointJson('DataSet.json')

print(keypoint.shape)
print(label.shape)
print(label)