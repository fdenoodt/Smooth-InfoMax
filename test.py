from torch.utils.data import Dataset
import os
import os.path
import torchaudio
from collections import defaultdict
import torch
import numpy as np
import random

def default_loader(path):
    return torchaudio.load(path, normalize=False)

# mean = -1456218.7500
# std = 135303504.0
audio_length = 20480

path = "C:\\GitHub\\Smooth-InfoMax\\datasets\\LibriSpeech\\train-clean-100\\26\\496\\26-496-0000.flac"
audio, samplerate = default_loader(path)

assert (
        samplerate == 16000
), "Watch out, samplerate is not consistent throughout the dataset!"

# discard last part that is not a full 10ms
max_length = audio.size(1) // 160 * 160

start_idx = random.choice(
    np.arange(160, max_length - audio_length - 0, 160)
)

audio = audio[:, start_idx: start_idx + audio_length]


# Calculate the mean and std based on the actual audio data
# mean = audio.mean()
# std = audio.std()
#
#
# # normalize
# audio = (audio - mean) / std
#
# # save to wav
# # unnormalize
# audio = audio * std + mean
audio = audio.float()
torchaudio.save("x.wav", audio, 16000)