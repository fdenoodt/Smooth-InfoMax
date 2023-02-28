from torch.utils.data import Dataset
import os
import os.path
import torchaudio
from collections import defaultdict
import torch
import numpy as np
import random
import pathlib


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


def resample(audio, curr_samplerate=22050, new_samplerate=16000):
    audio = torchaudio.functional.resample(
        audio, orig_freq=curr_samplerate, new_freq=new_samplerate)
    return audio


def add_random_noise(waveform, noise_level=0.05):
    '''
    Generated via Chat gpt: Gaussian white noise
    The noise_level parameter in the add_noise_torch function corresponds to the amplitude of the random noise that will be added to the audio waveform. It is a value between 0 and 1, with a default value of 0.01. The higher the value, the more noise will be added to the waveform.
    For example, if noise_level is set to 0.01, it will add noise with amplitude at most 1% of the amplitude of the original waveform, while if noise_level is set to 0.5, it will add noise with amplitude at most 50% of the amplitude of the original waveform.
    It's important to note that, this is just one way of add noise, you can also use different type of noise, with different distribution or different amplitude values depending on the use case.


    The noise that is added by the add_noise_torch function is Gaussian white noise, which is noise that has a probability density function (PDF) equal to that of the normal distribution, also known as a Gaussian distribution or bell curve. The term "white" refers to the fact that the noise has equal power in all frequency bands, which is different from other types of noise like "pink" or "brown" noise.
    White noise is often used as a model for various kinds of random processes and has a flat power spectral density. It's useful in many applications like signal processing, audio and image processing, where adding randomness can help in the analysis or simulation of the system.
    In the add_noise_torch function, we are using the torch.randn_like(waveform) function to generate Gaussian white noise with the same shape as the input waveform. It generates random numbers from a standard normal distribution (mean=0, std=1) and then multiply it with the noise_level to scale the amplitude of the noise.
    '''
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise


class RandomBackgroundNoise:
    ''' 
    Add random background noise to audio signal. Based on: https://jonathanbgn.com/2021/08/30/audio-augmentation.html
    '''

    def __init__(self, target_sample_rate, noise_dir, min_snr_db=0, max_snr_db=15):
        self.target_sr = target_sample_rate
        self.min_snr_db = min_snr_db
        self.max_snr_db = max_snr_db

        if not os.path.exists(noise_dir):
            raise IOError(f'Noise directory `{noise_dir}` does not exist')
        # find all WAV files including in sub-folders:
        self.noise_files_list = list(pathlib.Path(noise_dir).glob('**/*.wav'))
        if len(self.noise_files_list) == 0:
            raise IOError(
                f'No .wav file found in the noise directory `{noise_dir}`')

    def get_random_file(self):
        random_noise_file = random.choice(self.noise_files_list)
        noise, noise_sr = torchaudio.load(random_noise_file)
        return noise, noise_sr

    def __call__(self, audio_signal):
        c, audio_length = audio_signal.shape

        while True:
            noise, noise_sr = self.get_random_file()

            # Change sample rate to target
            noise = resample(noise, noise_sr, self.target_sr)
            c_noise, noise_length = noise.shape

            # change length
            if noise_length > audio_length:
                offset = random.randint(0, noise_length-audio_length)
                noise = noise[..., offset:offset+audio_length]
                break
            else:
                print("new attempt")
            # elif noise_length < audio_length:
            #     raise Exception("noise length should be longer than audio length")
                # noise = torch.cat([noise, torch.zeros((noise.shape[0], audio_length-noise_length))], dim=-1)

        snr_db = random.randint(self.min_snr_db, self.max_snr_db)
        speech_rms = audio_signal.norm(p=2)
        noise_rms = noise.norm(p=2)
        snr = 10 ** (snr_db / 20)
        scale = snr * noise_rms / speech_rms
        noisy_audio_speech = (scale * audio_signal + noise) / 2
        return noisy_audio_speech


class DeBoerDataset(Dataset):
    ''' Corpus of vocals consiting of three syllables per file, spoken by the same speaker. '''

    def __init__(
        self,
        opt,
        root,
        directory="train",
        # Resulting sequences will be of 16khz -> 16k samples per second
        # 16,000 samples per sec
        # 160 samples = 0.01 sec (10 ms)
        # 8821 // 160 = 55
        audio_length=55 * 160,  # -> 8800 elements # 64 is used in subsample in InfoNCE loss
        loader=default_loader,
        background_noise=False,
        white_guassian_noise=False,
        target_sample_rate=16000,
        background_noise_path=None
    ):
        self.root = root
        self.opt = opt
        self.target_sample_rate = target_sample_rate
        self.background_noise = background_noise
        self.white_guassian_noise = white_guassian_noise

        files = os.listdir(f"{root}/{directory}")
        # the Nones correspond to speaker_id and dir_id --> see default flist reader
        self.file_list = [(directory, fname.split(".wav")[0])
                          for fname in files]

        self.loader = loader
        self.audio_length = audio_length

        self.white_gaussian_noise_transform = add_random_noise
        self.noise_transform = RandomBackgroundNoise(
            self.target_sample_rate, background_noise_path) if background_noise else None

        # if background_noise flag is enabled, must also provide a path
        assert (not(background_noise_path is None)
                and background_noise) or not(background_noise)

    def __getitem__(self, index):
        dir_id, filename = self.file_list[index]
        # eg: filename = bagigi_1_1_ba

        full_word = filename.split("_")[0]  # bagigi
        pronounced_syllable = filename[-2:] # ba

        audio, samplerate = self.loader(
            os.path.join(self.root, dir_id, f"{filename}.wav"))

        audio_length_before_resample = audio.size(1)
        assert (
            samplerate == 22050
        ), "Watch out, samplerate is not consistent throughout the dataset!"

        assert (  # check only relevant for split up/padded audio files
            audio_length_before_resample == 12156  # computed in padding.py
        ), "Audio length is not consistent throughout the dataset!"

        # resample: from 22050 to 16000
        audio = resample(audio,
                         curr_samplerate=22050,
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



