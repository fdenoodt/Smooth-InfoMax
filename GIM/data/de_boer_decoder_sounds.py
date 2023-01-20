from torch.utils.data import Dataset
# from de_boer_sounds import DeBoerDataset

from data import de_boer_sounds

import os
import os.path
import torchaudio
from collections import defaultdict
import numpy as np
import random

from models.GIM_decoder import GIM_Encoder


def default_loader(path):
    return torchaudio.load(path)


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


class DeBoerDecoderDataset(de_boer_sounds.DeBoerDataset):
    def __init__(self, opt, root, directory="train", audio_length=64 * 160, loader=default_loader, background_noise=False, white_guassian_noise=False, target_sample_rate=16000, background_noise_path=None):
        super().__init__(opt, root, directory, audio_length, loader,
                         background_noise, white_guassian_noise, target_sample_rate, background_noise_path)

        self.encoder = GIM_Encoder(opt)

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
        audio = de_boer_sounds.resample(audio,
                                        curr_samplerate=44100,
                                        new_samplerate=16000)

        # discard last part that is not a full 10ms
        max_length = (audio.size(1) // 160) * 160

        start_idx = random.choice(
            np.arange(160, max_length - self.audio_length - 0, 160)
        )

        audio = audio[:, start_idx: start_idx + self.audio_length]

        # if encode via GIM is requested
        audio_enc = self.encoder(audio)

        # eg: [1, 2047, 512] -> [2047, 512] -> [512, 2047]
        audio_enc = audio_enc.squeeze(dim=0)
        audio_enc = audio_enc.permute(1, 0)
        audio_enc = audio_enc.cpu() # this fixes warning

        return audio, audio_enc, filename, speaker_id, start_idx
