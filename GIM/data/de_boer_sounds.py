from torch.utils.data import Dataset
import os
import os.path
import torchaudio
from collections import defaultdict
import torch
import numpy as np
import random


def default_loader(path):
    return torchaudio.load(path,
                           # normalization=False todo
                           )


def default_flist_reader(flist):
    item_list = []
    speaker_dict = defaultdict(list)
    index = 0
    with open(flist, "r") as rf:
        for line in rf.readlines():
            speaker_id, dir_id, sample_id = line.replace("\n", "").split("-")
            item_list.append((speaker_id, dir_id, sample_id))
            speaker_dict[speaker_id].append(index)
            index += 1

    return item_list, speaker_dict


class DeBoerDataset(Dataset):
    def __init__(
        self,
        opt,
        root,
        directory="train",
        # Resulting sequences will be of 16khz -> 16k samples per second
        # 16.000 samples per sec
        # 160 samples = 0.01 sec
        # 128 * 0.01 sec -> 1.28 second samples
        audio_length=64 * 160, # -> 20480 elements over 1.28 seconds
        # audio_length=128 * 160, # -> 20480 elements over 1.28 seconds
        loader=default_loader,
    ):
        self.root = root
        self.opt = opt

        files = os.listdir(f"{root}/{directory}")
        # the Nones correspond to speaker_id and dir_id --> see default flist reader
        self.file_list = [(0, directory, fname.split(".wav")[0])
                          for fname in files]

        self.loader = loader
        self.audio_length = audio_length

    def __getitem__(self, index):
        speaker_id, dir_id, sample_id = self.file_list[index]
        filename = f"{sample_id}"
        audio, samplerate = self.loader(
            os.path.join(self.root, dir_id, f"{filename}.wav")
        )

        assert (
            samplerate == 44100  # todo: before 16000 = 16Khz
        ), "Watch out, samplerate is not consistent throughout the dataset!"

        # resample
        audio = self.resample(audio,
                              curr_samplerate=44100,
                              new_samplerate=16000)

        # discard last part that is not a full 10ms
        max_length = (audio.size(1) // 160) * 160

        start_idx = random.choice(
            np.arange(160, max_length - self.audio_length - 0, 160)
        )

        # OLD = audio  # audio org: [1, 41278] -> [1, 20480]
        audio = audio[:,  # corresponds to first dim (= 1)
                      start_idx: start_idx + self.audio_length]

        return audio, filename, speaker_id, start_idx

    def __len__(self):
        return len(self.file_list)

    def resample(self, audio, curr_samplerate=44100, new_samplerate=16000):
        new_samplerate = 16000
        audio = torchaudio.functional.resample(
            audio, orig_freq=curr_samplerate, new_freq=new_samplerate)
        return audio

    def get_audio_by_speaker(self, speaker_id, batch_size=20):
        """
        get audio samples based on the speaker_id
        used for plotting the latent representations of different speakers
        """
        batch_size = min(len(self.speaker_dict[speaker_id]), batch_size)
        batch = torch.zeros(batch_size, 1, self.audio_length)
        for idx in range(batch_size):
            batch[idx, 0, :], _, _, _ = self.__getitem__(
                self.speaker_dict[speaker_id][idx]
            )

        return batch

    def get_full_size_test_item(self, index):
        """
        get audio samples that cover the full length of the input files
        used for testing the phone classification performance
        """

        speaker_id, dir_id, sample_id = self.file_list[index]
        filename = "{}-{}-{}".format(speaker_id, dir_id, sample_id)
        audio, samplerate = self.loader(
            os.path.join(self.root, speaker_id, dir_id,
                         "{}.flac".format(filename))
        )

        assert (
            samplerate == 44100
        ), "Watch out, samplerate is not consistent throughout the dataset!"

        # resample
        audio = self.resample(audio,
                              curr_samplerate=44100,
                              new_samplerate=16000)

        # discard last part that is not a full 10ms
        max_length = audio.size(1) // 160 * 160
        audio = audio[:max_length]

        audio = (audio - self.mean) / self.std

        return audio, filename
