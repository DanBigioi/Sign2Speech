import matplotlib.pyplot as plt
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

def plot_pose(ax, pose, plot_obj=True):
    # Plot the hand first:
    prev = pose[0, :]
    for row in np.ndindex(pose.shape[0] - 8):
        cur = pose[row, :]
        if row[0] in [5, 9, 13, 17]:
            prev = pose[0, :]
        cur, prev = cur.flatten(), prev.flatten()
        x, y, z = (
            np.linspace(prev[0], cur[0], 100),
            np.linspace(prev[1], cur[1], 100),
            np.linspace(prev[2], cur[2], 100),
        )
        ax.plot(x, y, z, color="red")
        ax.text(cur[0], cur[1], cur[2], f"{row[0]}", color="red")
        prev = cur
    if plot_obj:
        # Plot the object bbox:
        faces = [
            [0, 1, 2, 3, 0],
            [0, 1, 5, 4, 0],
            [0, 3, 7, 4, 0],
            [1, 5, 6, 2, 1],
            [2, 3, 7, 6, 2],
            [5, 6, 7, 4, 5]
        ]
        for face in faces:
            prev = pose[21+face[0], :]
            for idx in face:
                row = 21 + idx
                cur = pose[row, :]
                cur, prev = cur.flatten(), prev.flatten()
                x, y, z = (
                    np.linspace(prev[0], cur[0], 100),
                    np.linspace(prev[1], cur[1], 100),
                    np.linspace(prev[2], cur[2], 100),
                )
                ax.plot(x, y, z, color="green")
                ax.text(cur[0], cur[1], cur[2], f"{row-21}", color="green")
                prev = cur

#     scaling = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
#     ax.auto_scale_xyz(*[[np.min(scaling), np.max(scaling)]]*3)
    world_limits = ax.get_w_lims()
    ax.set_box_aspect((world_limits[1]-world_limits[0],world_limits[3]-world_limits[2],world_limits[5]-world_limits[4]))


def plot_3D_hand(pose):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_pose(ax, pose, plot_obj=False)
    plt.show()

