path = "C:\\GitHub\\Smooth-InfoMax\\datasets\\corpus\\reshuffled\\train"

# Get the mean and standard deviation the dataset.
# This is used to normalize the dataset. Consists of audio files (.wav)

import os
import numpy as np
import torchaudio

# Get all the files in the directory
files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

print(f"Number of files: {len(files)}")


# Get the mean and standard deviation of the dataset
mean = 0
std = 0
num_samples = 0

for file in files:
    # Load the audio file
    waveform, sample_rate = torchaudio.load(os.path.join(path, file))
    mean += waveform.mean()
    std += waveform.std()
    num_samples += 1

mean /= num_samples
std /= num_samples

print(f"Mean: {mean}, Standard Deviation: {std}")
# Mean: 3.260508094626857e-07, Standard Deviation: 0.10727367550134659
