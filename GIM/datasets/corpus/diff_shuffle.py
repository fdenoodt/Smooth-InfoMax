import shutil
import random
import os
directory = "./split up data cropped/test/"

# get file list in directory


def rename(directory):
    for filename in os.listdir(directory):
        # eg: filename = 'babidi_1_1_ba.wav', rename to 'babidi_2_1_ba.wav
        newfilename = filename.split('_')
        newfilename[1] = str(int(newfilename[1]) + 1)
        newfilename = '_'.join(newfilename)
        os.rename(os.path.join(directory, filename), os.path.join(directory, newfilename))

# rename(directory)



# Set the directory containing the .wav files
directory = "all"

# Set the training and testing directories
train_dir = "./split up data cropped reshuffled/train/"
test_dir = "./split up data cropped reshuffled/test/"

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
