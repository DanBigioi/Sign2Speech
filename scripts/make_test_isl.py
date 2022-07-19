#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2022 Théo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.

"""

"""

import os
import cv2
import argparse
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands



def interp_func(input_mat, src_fps=30, trg_fps=101):
    xp = list(np.arange(0, input_mat.shape[0], 1))
    interp_xp = list(np.arange(0, input_mat.shape[0], src_fps / trg_fps))
    interp_mat = np.zeros(shape=(len(interp_xp), input_mat.shape[1]))
    for j in range(input_mat.shape[1]):
        interp_mat[:, j] = np.interp(interp_xp, xp, input_mat[:, j])
    return interp_mat


def process_video(video_path) -> np.ndarray:
    poses = []
    cap = cv2.VideoCapture(video_path)
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6,
        static_image_mode=False,
        max_num_hands=1) as hands:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty vid frame.")
          # If loading a video, use 'break' instead of 'continue'.
          break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        pose = []
        if results.multi_hand_landmarks is None:
            continue
        for hand_landmarks in results.multi_hand_landmarks:
            for i, kp in enumerate(hand_landmarks.landmark):
                pose += [kp.x, kp.y, kp.z]
        poses.append(pose)
        # poses.append(
        # Draw the hand annotations on the image.
#         image.flags.writeable = True
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # if results.multi_hand_landmarks:
          # for hand_landmarks in results.multi_hand_landmarks:
            # mp_drawing.draw_landmarks(
                # image,
                # hand_landmarks,
                # mp_hands.HAND_CONNECTIONS,
                # mp_drawing_styles.get_default_hand_landmarks_style(),
                # mp_drawing_styles.get_default_hand_connections_style())
        # # Flip the image horizontally for a selfie-view display.
        # cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
        # if cv2.waitKey(5) & 0xFF == 27:
#           break
    cap.release()
    assert len(poses) >= 10
    if len(poses) >= 101:
        poses = np.array(poses[:101])
    else:
        poses = interp_func(np.array(poses), len(poses), 101)
    return poses


def process_dataset(input_path, output_path):
    n = 0
    for root, dir, files in os.walk(input_path):
        for file in files:
            fpath = os.path.join(root, file)
            if os.path.isfile(fpath):
                print(f"[*] Processing {fpath}...")
                vid_poses = process_video(fpath)
                print(vid_poses.shape)
                label = file[0].upper()
                os.makedirs(os.path.join(output_path, label), exist_ok=True)
                np.save(os.path.join(output_path, label, f"{n}.npy"), vid_poses)
                n += 1
    print(f"[*] Done! Processed {n} videos.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("dataset_dest")

    args = parser.parse_args()
    process_dataset(args.dataset_path, args.dataset_dest)
