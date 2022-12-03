# %%
import numpy as np
import torch
import random
import os
import torchaudio
import matplotlib.pyplot as plt


def compute_mean(path):
    fname = path + "/8419-286667-0008.flac"
    return torchaudio.load(fname)


def default_loader(path):
    return torchaudio.load(path, normalize=False)


def default_flist_reader(flist):
    item_list = []
    index = 0
    with open(flist, "r") as rf:
        for line in rf.readlines():
            speaker_id, dir_id, sample_id = line.replace("\n", "").split("-")
            item_list.append((speaker_id, dir_id, sample_id))
            # speaker_dict[speaker_id].append(index)
            index += 1

    return item_list


# def print_stats(waveform, sample_rate=None, src=None):
#     if src:
#         print("-" * 10)
#         print("Source:", src)
#         print("-" * 10)
#     if sample_rate:
#         print("Sample Rate:", sample_rate)
#     print("Shape:", tuple(waveform.shape))
#     print("Dtype:", waveform.dtype)
#     print(f" - Max:     {waveform.max().item():6.3f}")
#     print(f" - Min:     {waveform.min().item():6.3f}")
#     print(f" - Mean:    {waveform.mean().item():6.3f}")
#     print(f" - Std Dev: {waveform.std().item():6.3f}")
#     print()
#     print(waveform)
#     print()


audio, samplerate = compute_mean(
    "C:/GitHub/thesis-fabian-denoodt/GIM/datasets/LibriSpeech/train-clean-100/8419/286667")

print(audio)
print(audio.shape)
print(samplerate)

# %%
# print_stats(audio, samplerate)


root = "C:/GitHub/thesis-fabian-denoodt/GIM/datasets/LibriSpeech/train-clean-100/"

file_list = default_flist_reader(
    "C:/GitHub/thesis-fabian-denoodt/GIM/datasets/LibriSpeech100_labels_split/test_split.txt")
# "C:/GitHub/thesis-fabian-denoodt/GIM/datasets/LibriSpeech100_labels_split/train_split.txt")
random.shuffle(file_list)  # shuffle method

print(f"Length: {len(file_list)}")
total_sum = 0
total_duration = 0
for idx, file in enumerate(file_list):
    if idx % 100 == 0 and total_duration != 0:
        print(f"{idx}/{len(file_list)}")
        print(total_sum/(total_duration))

    speaker_id, dir_id, sample_id = file
    filename = "{}-{}-{}".format(speaker_id, dir_id, sample_id)
    full_name = os.path.join(root, speaker_id, dir_id, f"{filename}.flac")
    full_name = full_name.replace("/", "/")
    audio, samplerate = default_loader(full_name)

    audio_numpy = audio[0].to('cpu').numpy()

    duration = audio_numpy.shape[0]
    # _, duration = audio.shape

    total_sum += np.sum(audio_numpy) 
    # audio.sum()
    total_duration += duration


# train set: -0.0007
# test set: -0.0006
# %%

# torchaudio.load("C:/GitHub/thesis-fabian-denoodt/GIM/datasets/LibriSpeech/train-clean-100/5750/100289/5750-100289-0025.flac",
#                 normalize=False
#                 )


# %%
audio, _ = default_loader(
    "C:/GitHub/thesis-fabian-denoodt/GIM/datasets/gigabo/train/bababi_1.wav")

print(audio[0])
print(audio.shape)

# print(torch.mean(audio[0]))
# print(audio.max().item())

#
plt.plot(audio[0])
np.mean(audio[0].to('cpu').numpy())
