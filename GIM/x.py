# %%

import torch.nn as nn
import IPython.display as ipd
from ipywidgets import interact, fixed, FloatSlider
import librosa
from matplotlib import pyplot as plt
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
# %%

signal1 = torch.rand((64, 1, 10240))
signal_sin = torch.sin(torch.arange(0, 10240/1000, 1/1000)
                       ).repeat(64).reshape((64, 1, 10240))

print(signal1.shape)
print(signal_sin.shape)

plt.plot(signal_sin[0][0].to('cpu').detach().numpy())
plt.show()

torch.stft(signal_sin[0][0], )
# freq = torch.fft.fft(input=signal_sin[0][0])
freq.shape

# short_freq = signal_sin[0][0].stft(n_fft=195) # size of fourier transform
# short_freq.shape

plt.plot(freq.to('cpu').detach().numpy())
plt.show()


# %%

# chat gpt:


# Generate a sine wave
sample_rate = 44100
freq = 1200
num_samples = sample_rate
time = np.linspace(0, 1, num_samples, endpoint=False)
sine_wave = torch.tensor(np.sin(2 * np.pi * freq * time), dtype=torch.float32)

# Compute the STFT
window_size = 2048
hop_length = window_size // 4
stft = torch.stft(sine_wave, n_fft=window_size, hop_length=hop_length)

# Compute the magnitude and phase of the STFT
magnitude, phase = stft.abs(), stft.angle()

# Plot the original sine wave and the magnitude of the STFT
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(time, sine_wave.numpy())
plt.xlabel("Time (s)")
plt.title("Original Sine Wave")

plt.subplot(2, 1, 2)
plt.imshow(magnitude[:, 0, :].numpy(), origin='lower', aspect='auto',
           extent=[0, window_size / 2, 0, sample_rate / 2])
plt.xlabel("Frequency (Hz)")
plt.title("Magnitude of STFT")
plt.show()


# %%

Fs = 128
duration = 10
omega1 = 1
omega2 = 5
N = int(duration * Fs)
t = torch.arange(N) / Fs
t1 = t[:N//2]
t2 = t[N//2:]

x1 = 1.0 * torch.sin(2 * np.pi * omega1 * t1)
x2 = 0.7 * torch.sin(2 * np.pi * omega2 * t2)
x = torch.concatenate((x1, x2))

plt.figure(figsize=(8, 2))
plt.subplot(1, 2, 1)
plt.plot(t, x, c='k')
plt.xlim([min(t), max(t)])
plt.xlabel('Time (seconds)')

plt.subplot(1, 2, 2)
X = torch.abs(torch.fft.fft(x)) / Fs
freq = torch.fft.fftfreq(N, d=1/Fs)
# X = X[:N//2]
# freq = freq[:N//2]

plt.plot(freq.numpy(), X, c='k')
plt.xlim([0, 7])
plt.ylim([0, 3])
plt.xlabel('Frequency (Hz)')
plt.tight_layout()


# %%

def windowed_ft(t, x, Fs, w_pos_sec, w_len):

    N = len(x)
    w_pos = int(Fs * w_pos_sec)
    w_padded = np.zeros(N)
    w_padded[w_pos:w_pos + w_len] = 1
    x = x * w_padded
    plt.figure(figsize=(8, 2))

    plt.subplot(1, 2, 1)
    plt.plot(t, x, c='k')
    plt.plot(t, w_padded, c='r')
    plt.xlim([min(t), max(t)])
    plt.ylim([-1.1, 1.1])
    plt.xlabel('Time (seconds)')

    plt.subplot(1, 2, 2)
    X = np.abs(np.fft.fft(x)) / Fs
    freq = np.fft.fftfreq(N, d=1/Fs)
    X = X[:N//2]
    freq = freq[:N//2]
    plt.plot(freq, X, c='k')
    plt.xlim([0, 7])
    plt.ylim([0, 3])
    plt.xlabel('Frequency (Hz)')
    plt.tight_layout()
    plt.show()


w_len = 4 * Fs
windowed_ft(t, x, Fs, w_pos_sec=1, w_len=w_len)
windowed_ft(t, x, Fs, w_pos_sec=3, w_len=w_len)
windowed_ft(t, x, Fs, w_pos_sec=5, w_len=w_len)

print('Interactive interface for experimenting with different window shifts:')
interact(windowed_ft,
         w_pos_sec=FloatSlider(min=0, max=duration-(w_len/Fs), step=0.1,
                               continuous_update=False, value=1.7, description='Position'),
         t=fixed(t), x=fixed(x), Fs=fixed(Fs), w_len=fixed(w_len))


# %%

def stft_basic(x, w, H=8, only_positive_frequencies=False):
    """Compute a basic version of the discrete short-time Fourier transform (STFT)

    Notebook: C2/C2_STFT-Basic.ipynb

    Args:
        x (np.ndarray): Signal to be transformed
        w (np.ndarray): Window function
        H (int): Hopsize (Default value = 8)
        only_positive_frequencies (bool): Return only positive frequency part of spectrum (non-invertible)
            (Default value = False)

    Returns:
        X (np.ndarray): The discrete short-time Fourier transform
    """
    N = len(w)
    L = len(x)
    M = np.floor((L - N) / H).astype(int) + 1
    X = np.zeros((N, M), dtype='complex')
    for m in range(M):
        x_win = x[m * H:m * H + N] * w
        X_win = np.fft.fft(x_win)
        X[:, m] = X_win

    if only_positive_frequencies:
        K = 1 + N // 2
        X = X[0:K, :]
    return X


H = 8
N = 128
w = np.ones(N)
X = stft_basic(x, w, H, only_positive_frequencies=True)
Y = np.abs(X) ** 2

plt.figure(figsize=(8, 2))
plt.subplot(1, 2, 1)
plt.plot(np.arange(len(t)), x, c='k')
plt.xlim([0, len(t)])
plt.xlabel('Index (samples)')
plt.subplot(1, 2, 2)
plt.imshow(Y, origin='lower', aspect='auto', cmap='gray_r')
plt.xlabel('Index (frames)')
plt.ylabel('Index (frequency)')
plt.tight_layout()


# # %%
# import torch
# # # from torch.utils.data import Dataset
# # # # from de_boer_sounds import DeBoerDataset

# # # from data import de_boer_sounds

# # # import os
# # # import os.path
# # # import torchaudio
# # # from collections import defaultdict
# # # import numpy as np
# # # import random


# # # from models import full_model
# # # from utils import model_utils

# # from models.GIM_encoder import GIM_Encoder

# # # from models.GIM_encoder import GIM_Encoder

# # # encoder = GIM_Encoder(opt, layer_depth, GIM_encoder_path)

# # # data = torch.load("C:\\GitHub\\thesis-fabian-denoodt\\GIM\\g_drive_model\\model_180.ckpt",
# # #                                         map_location=torch.device('cuda')
# # #                                         #  map_location=device
# # #                                          )
# # # data

# # # # %%
# # # data.shape

# # # # x = GIM_Encoder()


# from decoder_architectures import TwoLayerDecoder

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# decoder = TwoLayerDecoder().to(device)

# model_path = "./logs\\\RMSE_decoder_GIM_layer3_experiment\\model_19.ckpt"
# decoder.load_state_dict(torch.load(model_path, map_location=device))

# rnd = torch.rand((2, 512, 256)).to('cuda')
# outp2 = decoder(rnd)
# outp2


# # %%

# %%


# In signal processing, the Fourier Transform (FT) is a technique for transforming a time-domain signal into its frequency components. The most common implementation of the FT is the Discrete Fourier Transform (DFT), which is the discrete version of the FT that can be applied to sampled signals. The DFT is computed using the Fast Fourier Transform (FFT) algorithm, which is an efficient algorithm for computing the DFT.
# The FFT is implemented as an efficient algorithm that uses the properties of the complex exponentials. The FFT algorithm requires the number of samples in the input signal to be a power of 2, which is not always the case for real-world signals. To overcome this limitation, a technique called zero-padding is used to pad the input signal with zeroes, so that it has a length that is a power of 2.
# n_fft is the parameter that determines the number of FFT bins used in the torch.stft() function for computing the STFT. The number of FFT bins is equal to the number of samples in the input signal after zero-padding. When you set n_fft to a value of 1024, the torch.stft() function will zero-pad the input signal to a length of 1024 before computing the STFT.
# In general, the larger the n_fft the more frequency resolution you will have, but it will also mean that the window used for the STFT will be wider and it can affect the time resolution of the STFT. Therefore, you need to choose the right value of n_fft depending on your specific task and dataset.


# Sure, I'd be happy to explain what n_fft is.
# In signal processing, the Fourier Transform (FT) is a technique for transforming a time-domain signal 
# into its frequency components. The most common implementation of the FT is the Discrete Fourier Transform (DFT), 
# which is the discrete version of the FT that can be applied to sampled signals. The DFT is computed using the 
# Fast Fourier Transform (FFT) algorithm, which is an efficient algorithm for computing the DFT.
# The FFT is implemented as an efficient algorithm that uses the properties of the complex exponentials. The FFT 
# algorithm requires the number of samples in the input signal to be a power of 2, which is not always the case for 
# real-world signals. To overcome this limitation, a technique called zero-padding is used to pad the input signal 
# with zeroes, so that it has a length that is a power of 2.
# n_fft is the parameter that determines the number of FFT bins used in the torch.stft() function for computing the STFT. 
# The number of FFT bins is equal to the number of samples in the input signal after zero-padding. When you set n_fft 
# to a value of 1024, the torch.stft() function will zero-pad the input signal to a length of 1024 before computing the STFT.
# In general, the larger the n_fft the more frequency resolution you will have, but it will also mean that the window 
# used for the STFT will be wider and it can affect the time resolution of the STFT. Therefore, you need to choose 
# the right value of n_fft depending on your specific task and dataset.

# %%


# generated via chat gpt.
# class SpectralLoss(nn.Module):
#     def __init__(self, n_fft):
#         # n_fft, which is the number of FFT bins to be used when performing the STFT. The forward method applies the STFT to the input and target speech signals using the torch.stft() function, and computes the power spectrogram of each signal by summing the squared magnitude of the STFT coefficients. It then applies the MSE loss between the two power spectrograms.
#         super(SpectralLoss, self).__init__()
#         self.n_fft = n_fft
#         self.loss = nn.MSELoss()

#     def forward(self, input, target):
#         input_spect  = torch.stft(input, self.n_fft, return_complex=False) # only magnitude
#         target_spect = torch.stft(target, self.n_fft, return_complex=False) # only magnitude
#         input_spect = input_spect.pow(2).sum(-1)
#         target_spect = target_spect.pow(2).sum(-1)
#         return self.loss(input_spect, target_spect)
    

# class SpectralLoss(nn.Module):
#     def __init__(self, n_fft):
#         # n_fft, which is the number of FFT bins to be used when performing the STFT. The forward method applies the STFT to the input and target speech signals using the torch.stft() function, and computes the power spectrogram of each signal by summing the squared magnitude of the STFT coefficients. It then applies the MSE loss between the two power spectrograms.
#         super(SpectralLoss, self).__init__()
#         self.n_fft = n_fft
#         self.loss = nn.MSELoss()

#     def forward(self, input, target):
#         input_spect  = torch.stft(input, self.n_fft, return_complex=True)
#         target_spect = torch.stft(target, self.n_fft, return_complex=True)
#         input_spect = input_spect.pow(2).sum(-1)
#         target_spect = target_spect.pow(2).sum(-1)
#         return self.loss(input_spect, target_spect)




# # fix support batches higher dimensions:
# class SpectralLoss(nn.Module):
#     def __init__(self, n_fft):
#         super(SpectralLoss, self).__init__()
#         self.n_fft = n_fft
#         self.loss = nn.MSELoss()

#     def forward(self, input, target):
#         input_spect  = torch.stft(input, self.n_fft, return_complex=False) # only magnitude
#         target_spect = torch.stft(target, self.n_fft, return_complex=False) # only magnitude
#         input_spect = input_spect.pow(2).sum(-1)
#         target_spect = target_spect.pow(2).sum(-1)
#         return self.loss(input_spect, target_spect)




# # .repeat(64).reshape((64, 1, 10240))
# signal_sin = torch.sin(torch.arange(0, 10240/1000, 1/1000)).to('cuda')
# # .repeat(64).reshape((64, 1, 10240))
# signal_sin2 = torch.sin(torch.arange(0, 10240/5000, 1/5000)).to('cuda')

# signal_sin = signal_sin.unsqueeze(0).repeat(5, 1, 1)
# signal_sin2 = signal_sin2.unsqueeze(0).repeat(5, 1, 1)

# loss = SpectralLoss(n_fft=1024)
# loss(signal_sin, signal_sin2)



class SpectralLoss(nn.Module):
    def __init__(self, n_fft):
        super(SpectralLoss, self).__init__()
        self.n_fft = n_fft
        self.loss = nn.MSELoss()

    def forward(self, batch_inputs, batch_targets): 
        assert batch_inputs.shape == batch_targets.shape

        (batch_size, one, length) = batch_inputs.shape
        batch_inputs = batch_inputs.squeeze(1) # (batch_size, length)
        batch_targets = batch_targets.squeeze(1) # (batch_size, length)

        input_spectograms  = torch.stft(batch_inputs, self.n_fft, return_complex=False) # only magnitude
        target_spectograms = torch.stft(batch_targets, self.n_fft, return_complex=False) # only magnitude
        
        input_spectograms = input_spectograms.pow(2).sum(-1)
        target_spectograms = target_spectograms.pow(2).sum(-1)
        return self.loss(input_spectograms, target_spectograms)

batch_size = 5
signal_sin = torch.sin(torch.linspace(0, 10240/1000, 10240, device='cuda')).repeat(batch_size).view(batch_size, 1, 10240)
signal_sin2 = torch.sin(torch.linspace(0, 10240/5000, 10240, device='cuda')).repeat(batch_size).view(batch_size, 1, 10240)

print(signal_sin.shape)

loss = SpectralLoss(n_fft=1024)
print(loss(signal_sin, signal_sin2))





# how do i choose the right n_fft value?
# The choice of the value of n_fft in the STFT (Short-Time Fourier Transform) determines the number of frequency bins in the resulting spectrogram. A larger value of n_fft will result in a higher frequency resolution, but will also increase the computational complexity of the STFT.

# A common rule of thumb is to choose n_fft to be the next power of 2 greater than or equal to the window size used in the STFT. This is because the FFT algorithm, which is used to compute the STFT, is most efficient when the input size is a power of 2.

# Another rule of thumb is to choose n_fft to be at least 2 times the highest frequency that you want to analyze.

# In your case, if you want to analyze a signal with a highest frequency of 5kHz, then you should use an n_fft of at least 10,240. This will give you a frequency resolution of 44100/10240 = 4.3 Hz per bin.

# It's also important to note that the choice of the window size, hop size, and padding will also affect the results of the STFT. It's good practice to experiment with different values of these parameters to find the one that best suits your application.




# oFabian LoL
# why are you saying "in my case 5khz"?
# I apologize for the confusion, I was referencing the example you provided where you defined the signal_sin2 variable as torch.sin(torch.arange(0, 10240/5000, 1/5000)).to('cuda'), which generates a sine wave with a frequency of 5kHz. So I recommended an n_fft of at least 10,240 which is 2 times the highest frequency that you want to analyze in this case 5kHz. In general, when working with STFT, it's important to choose an appropriate value for n_fft based on the highest frequency you want to analyze and the window size used, it's also good practice to experiment with different values of these parameters to find the one that best suits your application.



# signal_sin.shape

# %%

import librosa
path = r"C:\GitHub\thesis-fabian-denoodt\GIM\invented_audios"
librosa.get_duration(filename=f'{path}/109.wav') # in seconds


# current audio blocks are of 0.02 seconds, so 20ms.
# current decoder: 52 --> 10240

# 10240 ~= 52 blocks * 20 ~= 1040 ms = 1.04 seconds

# I want blocks of 0.2 seconds, so 200ms.
# So 10240 ~= 5 blocks


