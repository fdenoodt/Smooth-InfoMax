import IPython.display as ipd
from options import OPTIONS as opt
from utils import model_utils
import torch
from models import full_model
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd

device = 'cuda' if torch.cuda.is_available() else 'cpu'
opt['batch_size'] = 8


def plot_spectrogram(signal, name):
    # plot_spectrogram(audios[0].to('cpu').numpy()[0], "bibaga")
    """Compute power spectrogram with Short-Time Fourier Transform and plot result."""
    spectrogram = librosa.amplitude_to_db(librosa.stft(signal))
    plt.figure(figsize=(20, 15))
    librosa.display.specshow(spectrogram, y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Log-frequency power spectrogram for {name}")
    plt.xlabel("Time")
    plt.show()


def load_model(path):
    # Code comes from: def load_model_and_optimizer()
    kernel_sizes = [10, 8, 4, 4, 4]
    strides = [5, 4, 2, 2, 2]
    padding = [2, 2, 2, 2, 1]
    enc_hidden = 512
    reg_hidden = 256

    calc_accuracy = False
    num_GPU = None

    # Initialize model.
    model = full_model.FullModel(
        opt,
        kernel_sizes=kernel_sizes,
        strides=strides,
        padding=padding,
        enc_hidden=enc_hidden,
        reg_hidden=reg_hidden,
        calc_accuracy=calc_accuracy,
    )

    # Run on only one GPU for supervised losses.
    if opt["loss"] == 2 or opt["loss"] == 1:
        num_GPU = 1

    model, num_GPU = model_utils.distribute_over_GPUs(
        opt, model, num_GPU=num_GPU)

    optimizer = torch.optim.Adam(model.parameters(), lr=opt['learning_rate'])
    model.load_state_dict(torch.load(path, 
        map_location=device
        ))

    return model, optimizer


def play_sound(audio):
    ipd.Audio(audio, rate=16000)


# def plot_fft():
#     wave = audios[0][0].to('cpu').numpy()
#     X = np.fft.fft(wave)
#     X_mag = np.absolute(X)
#     plt.figure(figsize=(18, 10))
#     plt.plot(X_mag)  # magnitude spectrum
#     plt.xlabel('Frequency (Hz)')


