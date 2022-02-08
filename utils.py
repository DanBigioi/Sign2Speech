import numpy as np
import json

from sklearn import preprocessing


def read_poses_json(json_path):

    with open(json_path) as f:
        data = json.load(f)
        keypoint_array = []
        label_array = []
        for i in data["pose"]:
            keypoint_array.append(i)

        for i in data["label"]:
            label_array.append(i[1])

        # Encoding the labels so that pytorch is able to trasnform them into tensors
        label_array = np.asarray(label_array)
        le = preprocessing.LabelEncoder()
        label_array = le.fit_transform(label_array)

        keypoint_array = np.asarray(keypoint_array)
        keypoint_array = np.reshape(
            keypoint_array, (320, 63)
        )  # TODO: Why are those values hard coded here?

    return keypoint_array, label_array


def load_mfcc(filepath):
    mfcc = np.load(filepath)
    print("mfcc shape = " + str(mfcc.shape))
    return mfcc


def load_landmarks(filepath):
    landmarks = np.load(filepath)
    print("landmarks shape = " + str(landmarks.shape))
    return landmarks
