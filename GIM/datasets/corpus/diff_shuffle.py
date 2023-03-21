directory = "./test_diff_shuffle"

# get file list in directory
import os

old_name = '_1.wav' # replace with the old file name extension
new_name = '_2.wav' # replace with the new file name extension

def rename(directory, old_name, new_name):
  for filename in os.listdir(directory):
      if filename.endswith(old_name):
          new_filename = filename.replace(old_name, new_name)
          os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))


import os
import random
import shutil

# Set the directory containing the .wav files
directory = "corpus_all_data"

# Set the training and testing directories
train_dir = "train_diff_shuffle"
test_dir = "test_diff_shuffle"

# Set the percentage of files to use for training
train_percent = 0.8

# Create the training and testing directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Get a list of all the .wav files in the directory
file_list = [f for f in os.listdir(directory) if f.endswith('.wav')]

# Shuffle the file list randomly
random.shuffle(file_list)

# Calculate the number of files to use for training and testing
num_train_files = int(len(file_list) * train_percent)
num_test_files = len(file_list) - num_train_files

# Copy the training files to the training directory
for filename in file_list[:num_train_files]:
    src = os.path.join(directory, filename)
    dst = os.path.join(train_dir, filename)
    shutil.copyfile(src, dst)

# Copy the testing files to the testing directory
for filename in file_list[num_train_files:]:
    src = os.path.join(directory, filename)
    dst = os.path.join(test_dir, filename)
    shutil.copyfile(src, dst)
