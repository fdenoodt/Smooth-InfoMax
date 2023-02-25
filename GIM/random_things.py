# %%
import numpy as np
import torch
import random
import os
import torchaudio
import matplotlib.pyplot as plt


# %%
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
    "C:/GitHub/thesis-fabian-denoodt/GIM/datasets/corpus/train/bababi_1.wav")

print(audio[0])
print(audio.shape)

# print(torch.mean(audio[0]))
# print(audio.max().item())

#
plt.plot(audio[0])
np.mean(audio[0].to('cpu').numpy())


# %%

waveform, sample_rate = torchaudio.load(r'C:/GitHub/thesis-fabian-denoodt/GIM/datasets/corpus/train/bababi_1.wav', normalize=True)
new = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)

print(waveform.shape)
print(new.shape)

waveform1 = waveform
waveform2 = new

# %%

# import matplotlib.pyplot as plt
# import numpy as np


# plt.figure(1)
# plt.title("Signal Wave...")
# plt.plot(waveform1[0])
# plt.plot(waveform2[0])
# plt.show()

# %%
waveform1.shape
waveform2.shape


# %%
# https://blog.deepgram.com/pytorch-intro-with-torchaudio/
# def print_stats(waveform, sample_rate=None, src=None):
#    if src:
#        print("-"*10)
#        print(f"Source: {src}")
#        print("-"*10)
#    if sample_rate:
#        print(f"Sample Rate: {sample_rate}")
#    print("Dtype:", waveform.dtype)
#    print(f" - Max:     {waveform.max().item():6.3f}")
#    print(f" - Min:     {waveform.min().item():6.3f}")
#    print(f" - Mean:    {waveform.mean().item():6.3f}")
#    print(f" - Std Dev: {waveform.std().item():6.3f}")
#    print()
#    print(waveform)
#    print()

# def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
#    waveform = waveform.numpy()
#    num_channels, num_frames = waveform.shape
#    figure, axes = plt.subplots(num_channels, 1)
#    if num_channels == 1:
#        axes = [axes]
#    for c in range(num_channels):
#        axes[c].specgram(waveform[c], Fs=sample_rate)
#        if num_channels > 1:
#            axes[c].set_ylabel(f"Channel {c+1}")
#        if xlim:
#            axes[c].set_xlim(xlim)
#    figure.suptitle(title)
#    plt.show(block=False)

# def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
#    waveform = waveform.to('cpu').numpy()
#    num_channels, num_frames = waveform.shape
#    time_axis = torch.arange(0, num_frames) / sample_rate

#    figure, axes = plt.subplots(num_channels, 1)
#    if num_channels == 1:
#        axes = [axes]
#    for c in range(num_channels):
#        axes[c].plot(time_axis, waveform[c], linewidth=1)
#        axes[c].grid(True)
#        if num_channels > 1:
#            axes[c].set_ylabel(f"Channel {c+1}")
#        if xlim:
#            axes[c].set_xlim(xlim)
#        if ylim:
#            axes[c].set_ylim(ylim)
#    figure.suptitle(title)
#    plt.show(block=False)

# print_stats(waveform1, sample_rate=None, src="Original")
# print_stats(waveform2, sample_rate=None, src="Effects Applied")
# plot_waveform(waveform1, None, title="Original", xlim=(-0.1, 3.2))
# plot_specgram(waveform1, None, title="Original", xlim=(0, 3.04))
# plot_waveform(waveform2, None, title="Effects Applied", xlim=(-0.1, 3.2))
# plot_specgram(waveform2, None, title="Effects Applied", xlim=(0, 3.04))