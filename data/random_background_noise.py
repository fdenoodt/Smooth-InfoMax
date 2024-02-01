import os
import random
import pathlib
import torchaudio
import torch

from utils.helper_functions import resample


class GuassianNoise:
    def add_random_noise(self, waveform, noise_level):
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

    def __call__(self, audio_signal):
        noise = self.add_random_noise(audio_signal, noise_level=0.005)
        return noise


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

        while True:  # try until we get a noise with the right length
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
                pass

        # Add noise to audio signal with random SNR between min_snr_db and max_snr_db
        snr_db = random.randint(self.min_snr_db, self.max_snr_db)
        speech_rms = audio_signal.norm(p=2)
        noise_rms = noise.norm(p=2)
        snr = 10 ** (snr_db / 20)
        scale = snr * noise_rms / speech_rms
        noisy_audio_speech = (scale * audio_signal + noise) / 2
        return noisy_audio_speech
