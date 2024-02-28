import torch
from torch.utils.data import Dataset
import os
import os.path
import torchaudio
from collections import defaultdict
from config_code.config_classes import DataSetConfig
from utils.helper_functions import resample, translate_syllable_to_number, translate_syllable_vowel_number


def default_loader(path):
    return torchaudio.load(path, normalize=True)


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
    ''' Corpus of vocals consisting of three syllables per file, spoken by the same speaker. '''

    def __init__(
            self,
            dataset_options: DataSetConfig,
            root,
            directory="train",
            loader=default_loader,
            target_sample_rate=16000,
    ):
        self.root = root
        self.opt = dataset_options
        self.target_sample_rate = target_sample_rate
        self.split_into_syllables = dataset_options.split_in_syllables
        self.initial_sample_rate = 22050 if self.split_into_syllables else 44100

        files = os.listdir(f"{root}/{directory}")
        # the Nones correspond to speaker_id and dir_id --> see default flist reader
        self.file_list = [(directory, fname.split(".wav")[0]) for fname in files]

        self.loader = loader
        self.audio_length: int = self.compute_audio_length()

        # # Mean: 3.260508094626857e-07, Standard Deviation: 0.10727367550134659
        # self.mean = 3.260508094626857e-07
        # self.std = 0.10727367550134659

    def compute_audio_length(self):
        # Resulting sequences will be of 16khz -> 16k samples per second
        # 16,000 samples per sec
        # 160 samples = 0.01 sec (10 ms)
        audio_length = 0
        if self.split_into_syllables:
            # 8821 // 160 = 55
            # 3452 // 160 = 21
            audio_length = 55 * 160  # -> 8800 elements
            # audio_length = 21 * 160  # -> 8800 elements

        else:
            # the length of the audio files is similar because syllables are padded with zeros in front and back
            audio_length = 64 * 160  # -> 10240 elements over 0.64 seconds
        return audio_length

    def __getitem__(self, index):
        dir_id, filename = self.file_list[index]
        # eg: filename = bagigi_1_1_ba if split, else filename = bagigi_1

        full_word = filename.split("_")[0]  # bagigi
        if self.split_into_syllables:
            pronounced_syllable = filename[-2:]  # ba
            if self.opt.labels == 'syllables':
                pronounced_syllable = translate_syllable_to_number(pronounced_syllable)  # 0
            else:
                pronounced_syllable = translate_syllable_vowel_number(pronounced_syllable)  # either 0, 1 or 2

        else:
            pronounced_syllable = 0  # dummy value as None is not supported by pytorch

        audio, samplerate = self.loader(
            os.path.join(self.root, dir_id, f"{filename}.wav"))
        audio = audio.float()

        audio_length_before_resample = audio.size(1)
        assert (
                samplerate == self.initial_sample_rate
        ), "Watch out, samplerate is not consistent throughout the dataset!"

        # resample: from 22050 to 16000
        audio = resample(audio,
                         curr_samplerate=self.initial_sample_rate,
                         new_samplerate=self.target_sample_rate)
        # length which originally was 12156 (all lengths are equal), are now 8821 due to lower samplerate

        # audio = audio[:, 0: self.audio_length]  # 10240 if not split, 8800 if split

        if not (self.split_into_syllables):
            audio = audio[:, 0: self.audio_length]
        else:
            # no need to cut the audio, as the data is already preprocessed
            pass

        # TODO
        # problem: only does the first part, but should consider a random starting point

        # store audio to file
        # torchaudio.save(f'{filename}.wav', audio, self.target_sample_rate)

        return audio, filename, pronounced_syllable, full_word

    def __len__(self):
        return len(self.file_list)
