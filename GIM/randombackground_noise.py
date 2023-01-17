# %%
import math
import os
import pathlib
import random
import torch
import torchaudio
from IPython.display import Audio


# %%


# batch = torch.load("audio_batch")


# waveform1 = batch[0]
# sample_rate1 = 16000
# Audio(waveform1, rate=sample_rate1)

# RMS = torch.sqrt(torch.mean(waveform1**2))
# STD_n = RMS
# # noise=np.random.normal(0, STD_n, waveform1.shape[0])
# noise = torch.randn(waveform1.shape[0]) * STD_n
# print(noise)

# # noise=np.random.normal(0, STD_n, waveform1.numpy().shape[0])
# # print(noise)

# signal_noise = waveform1+noise
# Audio(waveform1, rate=sample_rate1)








noise_transform = RandomBackgroundNoise(sample_rate1, './datasets/noise temp')
transformed_audio = noise_transform(batch[0])

n = add_noise(batch[0], 0.001)
Audio(n, rate=sample_rate1)
# Audio(transformed_audio, rate=sample_rate1)


# %%


def resample(audio, curr_samplerate=44100, new_samplerate=16000):
    new_samplerate = 16000
    audio = torchaudio.functional.resample(
        audio, orig_freq=curr_samplerate, new_freq=new_samplerate)
    return audio


batch = torch.load("audio_batch")
waveform1 = batch[0]
sample_rate1 = 16000

Audio(waveform1, rate=sample_rate1)
speech = waveform1

# speech, _ = torchaudio.load(SAMPLE_SPEECH)
noise, noise_sr = torchaudio.load("datasets/noise temp/music-rfm-0003.wav")
noise = resample(noise, noise_sr, sample_rate1)
noise = noise[:, : speech.shape[1]]

speech_rms = speech.norm(p=2)
noise_rms = noise.norm(p=2)

snr_dbs = [20, 10, 3]
noisy_speeches = []
for snr_db in snr_dbs:
    snr = 10 ** (snr_db / 20)
    print(snr)
    scale = snr * noise_rms / speech_rms
    noisy_speeches.append((scale * speech + noise) / 2)
# plot_waveform(noise, sample_rate, title="Background noise")
# plot_specgram(noise, sample_rate, title="Background noise")
# %%
Audio(noisy_speeches[0], rate=sample_rate1)
# %%
Audio(noisy_speeches[1], rate=sample_rate1)
# %%
Audio(noisy_speeches[2], rate=sample_rate1)
# %%
