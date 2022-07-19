import json
import numpy as np
from sklearn import preprocessing
import torch
import os
import librosa
import numpy as np
from scipy.io.wavfile import read
import torch
import matplotlib as plt


def process_mfcc_from_audio(path_to_wav_dir, mfcc_dest_dir):
    os.chdir(path_to_wav_dir)
    for audio_file in os.listdir():
        print(audio_file)
        if audio_file.endswith('.wav'):
            signal, sr = librosa.load(path_to_wav_dir + '/' + audio_file, sr=None, mono=True)
            #signal = librosa.util.normalize(signal)
            n_fft = 1764
            hop_length = 441

            mel_spectrogram = librosa.feature.melspectrogram(y= signal, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                         n_mels=64)
            log_mel_spectrogram = librosa.power_to_db(mel_spectrogram,)
            log_mel_spectrogram = np.array(log_mel_spectrogram).T

            np.save(mfcc_dest_dir + '/' + audio_file[:-4] + '.npy', log_mel_spectrogram)

            
def interp_func(input_mat, src_fps=30, trg_fps=101):
    xp = list(np.arange(0, input_mat.shape[0], 1))
    interp_xp = list(np.arange(0, input_mat.shape[0], src_fps/trg_fps))
    interp_mat = np.zeros(shape=(len(interp_xp), input_mat.shape[1]))
    for j in range(input_mat.shape[1]):
        interp_mat[:, j] = np.interp(interp_xp, xp, input_mat[:, j])
    return interp_mat