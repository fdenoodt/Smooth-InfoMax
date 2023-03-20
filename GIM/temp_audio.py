# %%
import torchaudio.transforms as T
import torchaudio.functional as F
import torchaudio
import torch
import IPython.display as ipd
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
directory = r"C:\GitHub\thesis-fabian-denoodt\GIM\datasets\corpus\split up data padded\train"
directory = r"C:\GitHub\thesis-fabian-denoodt\GIM\datasets\corpus\train"

file = "bababi_1.wav"
signal, sr = librosa.load(f"{directory}/{file}", sr=16000)

file2 = "bababu_1.wav"
signal2, sr2 = librosa.load(f"{directory}/{file2}", sr=16000)


plt.figure(figsize=(20, 5))
librosa.display.waveplot(signal, sr=sr)
plt.title('Waveplot', fontdict=dict(size=18))
plt.xlabel('Time', fontdict=dict(size=15))
plt.ylabel('Amplitude', fontdict=dict(size=15))
plt.show()

ipd.Audio(signal, rate=sr)

# %%


def numpy_fft(signal):
    # Creating a Discrete-Fourier Transform with our FFT algorithm
    fast_fourier_transf = np.fft.fft(signal)
    # Magnitudes indicate the contribution of each frequency
    magnitude = np.abs(fast_fourier_transf)
    # mapping the magnitude to the relative frequency bins
    frequency = np.linspace(0, sr, len(magnitude))
    # We only need the first half of the magnitude and frequency
    left_mag = magnitude[:int(len(magnitude)/2)]
    left_freq = frequency[:int(len(frequency)/2)]

    return left_freq, left_mag


def plot(xs, ys):
    plt.plot(xs, ys)
    plt.title('Discrete-Fourier Transform', fontdict=dict(size=15))
    plt.xlabel('Frequency', fontdict=dict(size=12))
    plt.ylabel('Magnitude', fontdict=dict(size=12))
    plt.show()


xs, ys = numpy_fft(signal)
plot(xs, ys)
# print(signal.shape)
# print(fast_fourier_transf.shape)

# %%


def torch_fft(signal):
    s = torch.from_numpy(signal)
    fast_fourier_transf = torch.fft.fft(s)

    magnitude = torch.abs(fast_fourier_transf)
    frequency = torch.linspace(0, sr, len(magnitude))
    left_mag = magnitude[:int(len(magnitude)/2)]
    left_freq = frequency[:int(len(frequency)/2)]

    left_mag = left_mag.detach().numpy()
    left_freq = left_freq.detach().numpy()
    return left_freq, left_mag


xs, ys = torch_fft(signal)
plot(xs, ys)


# %%

def spectogram(signal, n_fft, hop_length):

    # Short-time Fourier Transformation on our audio data
    audio_stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)

    # gathering the absolute values for all values in our audio_stft
    spectrogram = np.abs(audio_stft)

    return spectrogram


def show_spectogram(spectrogram, hop_length):
    # Plotting the short-time Fourier Transformation
    plt.figure(figsize=(20, 5))
    # Using librosa.display.specshow() to create our spectrogram
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time',
                             y_axis='hz', hop_length=hop_length, cmap='magma')

    plt.colorbar(label='Amplitude (im1)/Decibels (im2)')
    plt.title('Spectrogram (amp)', fontdict=dict(size=18))
    plt.xlabel('Time', fontdict=dict(size=15))
    plt.ylabel('Frequency', fontdict=dict(size=15))
    plt.show()


# this is the number of samples in a window per fft
n_fft = 2048  # sample rate = 16000, so 2048 samples is 0.128 seconds

# The amount of samples we are shifting after each fft
hop_length = 512

spectogram = spectogram(signal, n_fft, hop_length)
show_spectogram(spectogram, hop_length)

# %%

log_spectro = librosa.amplitude_to_db(spectogram)
show_spectogram(log_spectro, hop_length)


mel_signal = librosa.feature.melspectrogram(y=signal,
                                            sr=sr,
                                            hop_length=hop_length,
                                            n_fft=n_fft)
spectrogram = np.abs(mel_signal)
power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
plt.figure(figsize=(8, 7))
librosa.display.specshow(power_to_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma',
                         hop_length=hop_length)
plt.colorbar(label='dB')
plt.title('Mel-Spectrogram (dB)', fontdict=dict(size=18))
plt.xlabel('Time', fontdict=dict(size=15))
plt.ylabel('Frequency', fontdict=dict(size=15))
plt.show()


# %%
# https://pytorch.org/audio/main/tutorials/audio_feature_extractions_tutorial.html#melspectrogram


def plot_waveform(waveform, sr, title="Waveform"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    figure, axes = plt.subplots(num_channels, 1)
    axes.plot(time_axis, waveform[0], linewidth=1)
    axes.grid(True)
    figure.suptitle(title)
    plt.show(block=False)


# n_fft = 1024
win_length = None
# hop_length = 512

# Define transform
spectrogram = T.Spectrogram(
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
)

# Perform transform
s = torch.from_numpy(signal)
spec = spectrogram(s)  # (channel, freq, time)
# plot_spectrogram(spec, title="torchaudio")


ss = torch.from_numpy(signal2)
ss = ss.unsqueeze(0)
s = s.unsqueeze(0)

print(ss.shape)

# %%

# stack the two signals
stacked = torch.stack([s, ss], dim=0)
print(stacked.shape)

# %%
n_mels = 128

mel_spectrogram = T.MelSpectrogram(
    sample_rate=sr,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    onesided=True,
    n_mels=n_mels,
    mel_scale="htk",
)

# melspec_librosa = librosa.feature.melspectrogram(
#     y=signal,
#     sr=sr,
#     n_fft=n_fft,
#     hop_length=hop_length,
#     win_length=win_length,
#     center=True,
#     pad_mode="reflect",
#     power=2.0,
#     n_mels=n_mels,
#     norm="slaney",
#     htk=True,
# )


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")

    print(specgram.shape)
    print(librosa.power_to_db(specgram).shape)

    im = axs.imshow(
        # specgram,
        librosa.power_to_db(specgram),
        origin="lower", aspect="auto"
    )
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


melspec = mel_spectrogram(stacked)

# amin is tensor of shape melspec, consisting of amin values
#amin = 1e-10
amin = 1e-10 * torch.ones_like(melspec)
ref_value = torch.ones_like(melspec)

log_spec = 10.0 * torch.log10(torch.maximum(amin, melspec))
log_spec -= 10.0 * torch.log10(torch.maximum(amin, ref_value))


plot_spectrogram(
    melspec[0][0], title="MelSpectrogram - librosa", ylabel="mel freq")
plot_spectrogram(
    melspec[1][0], title="MelSpectrogram - librosa", ylabel="mel freq")


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")

    print(specgram.shape)
    print(librosa.power_to_db(specgram).shape)

    im = axs.imshow(
        specgram,
        # librosa.power_to_db(specgram),
        origin="lower", aspect="auto"
    )
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


plot_spectrogram(
    log_spec[0][0], title="MelSpectrogram - librosa", ylabel="mel freq")
plot_spectrogram(
    log_spec[1][0], title="MelSpectrogram - librosa", ylabel="mel freq")


# melspec = mel_spectrogram(s)

# plot_spectrogram(melspec_librosa, title="MelSpectrogram - librosa", ylabel="mel freq")
# print(melspec.shape)

# mse = torch.square(melspec - melspec_librosa).mean().item()
# print("Mean Square Difference: ", mse)


# %%
