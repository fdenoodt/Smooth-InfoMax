from torch.utils.data import Dataset
import os
import os.path
import torchaudio
from collections import defaultdict

from data.random_background_noise import GuassianNoise, RandomBackgroundNoise
from helper_functions import resample, translate_syllable_to_number


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
    ''' Corpus of vocals consiting of three syllables per file, spoken by the same speaker. '''

    def __init__(
        self,
        opt,
        root,
        directory="train",
        loader=default_loader,
        background_noise=False,
        white_guassian_noise=False,
        target_sample_rate=16000,
        background_noise_path=None,
        split_into_syllables=False,
    ):
        self.root = root
        self.opt = opt
        self.target_sample_rate = target_sample_rate
        self.background_noise = background_noise
        self.white_guassian_noise = white_guassian_noise
        self.split_into_syllables = split_into_syllables
        self.initial_sample_rate = 22050 if split_into_syllables else 44100

        files = os.listdir(f"{root}/{directory}")
        # the Nones correspond to speaker_id and dir_id --> see default flist reader
        self.file_list = [(directory, fname.split(".wav")[0])
                          for fname in files]

        self.loader = loader
        self.audio_length: int = self.compute_audio_length()

        self.white_gaussian_noise_transform = GuassianNoise()
        self.noise_transform: RandomBackgroundNoise = RandomBackgroundNoise(
            self.target_sample_rate, background_noise_path) if background_noise else None

        # if background_noise flag is enabled, must also provide a path
        assert (not(background_noise_path is None)
                and background_noise) or not(background_noise)

    def compute_audio_length(self):
        # Resulting sequences will be of 16khz -> 16k samples per second
        # 16,000 samples per sec
        # 160 samples = 0.01 sec (10 ms)
        audio_length = 0
        if self.split_into_syllables:
            # 8821 // 160 = 55
            audio_length = 55 * 160  # -> 8800 elements
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
            pronounced_syllable = translate_syllable_to_number(
                pronounced_syllable)  # 0
        else:
            pronounced_syllable = 0  # dummy value as None is not supported by pytorch

        audio, samplerate = self.loader(
            os.path.join(self.root, dir_id, f"{filename}.wav"))

        audio_length_before_resample = audio.size(1)
        assert (
            samplerate == self.initial_sample_rate
        ), "Watch out, samplerate is not consistent throughout the dataset!"

        if self.split_into_syllables:
            assert (  # check only relevant for split up/padded audio files
                audio_length_before_resample == 12156  # computed in padding.py
            ), "Audio length is not consistent throughout the dataset!"

        # resample: from 22050 to 16000
        audio = resample(audio,
                         curr_samplerate=self.initial_sample_rate,
                         new_samplerate=self.target_sample_rate)
        # length which originally was 12156 (all lengths are equal), are now 8821 due to lower samplerate

        # Discard last part that is not a full 10ms
        audio = audio[:, 0: self.audio_length]  # resulting in 8800 elements

        if self.background_noise:
            audio = self.noise_transform(audio)

        if self.white_guassian_noise:
            audio = self.white_gaussian_noise_transform(audio)

        return audio, filename, pronounced_syllable, full_word

    def __len__(self):
        return len(self.file_list)
