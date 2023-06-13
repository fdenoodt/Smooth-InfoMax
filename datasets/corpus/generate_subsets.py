"""
The folder `split up data padded reshuffled` > `train` contains roughly 2000 files of the following format:
bidadi_1_1_bi.wav, where "bi" is the label. There are 9 labels: "ba", "bi", "bu", "da", "di", "du", "ga", "gi", "gu".

This script generates subsets of the training data, each with a different number of files per label.
The following subset sizes are generated:
    1) 1 files per label
    2) 2 files per label
    3) 4 files per label
    4) 8 files per label
    5) 16 files per label
    6) 32 files per label
    7) 64 files per label
    8) 128 files per label
    9) all files

The subsets are saved in the folder `subsets`. Within this folder, each subset is saved in a folder with the name of the subset size.
"""

import os
import shutil
import random

# Set the directory containing the .wav files
directory = "./split up data padded reshuffled/train/"

# Set the subset directory
subset_dir = "./subsets/"

# Create the subset directory
os.makedirs(subset_dir, exist_ok=True)

# Get a list of all the .wav files in the directory
file_list = [f for f in os.listdir(directory) if f.endswith('.wav')]
random.shuffle(file_list)

# Get a list of all the labels
labels = list(set([f.split('_')[-1].split('.')[0] for f in file_list])) # ['bi', 'di', 'gu', 'ga', 'du', 'da', 'bu', 'ba', 'gi']

# Get a list of all the files per label
files_per_label = {}
for label in labels:
    files_per_label[label] = [f for f in file_list if f.split('_')[-1].split('.')[0] == label]

# Generate the subsets
subset_sizes = [1, 2, 4, 8, 16, 32, 64, 128, len(file_list)]
for subset_size in subset_sizes:
    # Create the subset directory
    subset = os.path.join(subset_dir, str(subset_size))
    os.makedirs(subset, exist_ok=True)

    # Copy the subset files to the subset directory
    for label in labels:
        for filename in files_per_label[label][:subset_size]:
            src = os.path.join(directory, filename)
            dst = os.path.join(subset, filename)
            shutil.copyfile(src, dst)

