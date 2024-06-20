import os
import time
from typing import Any

import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import soundfile as sf
import torch
import torchaudio
from torchvision import transforms

try:
    import tikzplotlib  # some versions of Python have issues with this import
except:
    print("tikzplotlib not installed, will not be able to save as .tex")

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class EpochPrinter():
    def __init__(self, options, train_loader, learning_rate, criterion, decoder_depth=1) -> None:
        self.starttime = time.time()

        self.options = options
        self.print_idx = 100
        self.step = 0
        self.total_step = len(train_loader)
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.decoder_depth = decoder_depth

    def __call__(self, step, epoch) -> Any:
        opt = self.options
        if step % self.print_idx == 0:
            max_epochs = opt['num_epochs'] + opt.encoder_config.start_epoch
            print(
                f"Epoch[{epoch + 1}/{max_epochs}], Step[{step}/{self.total_step}], Time(s): {time.time() - self.starttime: .1f} L: {self.decoder_depth} lr: {self.learning_rate}, {self.criterion.name}")


def create_dir(path):
    create_log_dir(path)


def create_log_dir(path):  # created via chat gpt
    if not os.path.exists(path):
        os.makedirs(path)


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


def show_line_sequence(sequence, show=True, file=None):
    plt.plot(sequence.to('cpu').detach().numpy())

    if file is not None:
        plt.savefig(file)

    if show:
        plt.show()

    plt.clf()


def play_sound(audio):
    audio = audio.to('cpu').detach().numpy()
    ipd.Audio(audio, rate=16000)


def compute_normalizer(train_loader, encoder):
    sum = 0.0
    squared_sum = 0.0
    num_samples = 0

    # Iterate through the data using the data loader
    for step, (batch_audio_signals, _, _, _) in enumerate(train_loader):
        batch_audio_signals = batch_audio_signals.to(device)
        enc_audios_at_each_module = encoder(batch_audio_signals)
        enc_audios = enc_audios_at_each_module[0].to(
            device)  # (batch_size, 512, 256)

        b, c, l = enc_audios.shape
        enc_audios = enc_audios.reshape(b * l * c)

        # Add the sum and squared sum of the current batch to the running total
        sum += enc_audios.sum()
        squared_sum += (enc_audios ** 2).sum()
        num_samples += enc_audios.shape[0]

    # Compute the mean and standard deviation
    mean = sum / num_samples
    std = torch.sqrt(squared_sum / num_samples - mean ** 2)

    print('Mean:', mean)
    print('Standard deviation:', std)

    transform_norm = transforms.Compose([
        transforms.Normalize(mean, std)
    ])
    return transform_norm


def det_np(data):
    ''' convert tensor to numpy '''
    # detach + numpy
    return data.to('cpu').detach().numpy()


# def plot_fft():
#     wave = audios[0][0].to('cpu').numpy()
#     X = np.fft.fft(wave)
#     X_mag = np.absolute(X)
#     plt.figure(figsize=(18, 10))
#     plt.plot(X_mag)  # magnitude spectrum
#     plt.xlabel('Frequency (Hz)')


def fft_magnitude(sequence):
    ''' Compute the FFT magnitude of a sequence '''
    # return np.fft.fft(sequence)
    x = np.fft.fft(sequence)
    x_mag = np.absolute(x)
    x_mag = x_mag[:int(len(x_mag) / 2)]
    return x_mag


# if 16khz, only 8000 frequencies possible -> sample rate should be twice as large as the highest frequency


def plot_two_graphs_side_by_side(sequence1, sequence2, title="True vs Predicted", dir=None, file=None, show=True,
                                 fig_size=None, y_lims=None, type1="line", type2="line"):
    ''' Plot two graphs side by side '''
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)

    if type1 == "line":
        ax1.plot(sequence1)
    elif type1 == "bar":
        ax1.bar(np.arange(len(sequence1)), sequence1)

    if type2 == "line":
        ax2.plot(sequence2)
    elif type2 == "bar":
        ax2.bar(np.arange(len(sequence2)), sequence2)

    # set y axis limit
    if y_lims is not None:
        ax1.set_ylim(y_lims[0][0], y_lims[0][1])
        ax2.set_ylim(y_lims[1][0], y_lims[1][1])

    if file is not None:
        create_log_dir(dir)
        plt.savefig(f"{dir}/{file}")

    if fig_size is not None:
        fig.set_size_inches(fig_size[0], fig_size[1])

    if show:
        plt.show()

    plt.clf()


def plot_four_graphs_side_by_side(sequence1, sequence2, sequence3, sequence4, title="True vs Predicted", dir=None,
                                  file=None, show=True):
    ''' Plot four graphs side by side '''

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle(title)

    ax1.plot(sequence1)
    ax2.plot(sequence2)
    ax3.plot(sequence3)
    ax4.plot(sequence4)

    if file is not None:
        create_log_dir(dir)
        plt.savefig(f"{dir}/{file}")

    if show:
        plt.show()

    plt.clf()
    plt.cla()
    plt.close(fig)


def colour_palette():
    return np.array([np.array([1, 0.3712, 0.34]),  # ba
                     np.array([0.34, 0.34, 1]),  # bi
                     np.array([0.34, 1, 0.34]),  # bu
                     np.array([0.86, 0.34, 0.34]),  # da
                     np.array([0.34, 0.34, 0.86]),  # di
                     np.array([0.34, 0.86, 0.34]),  # du
                     np.array([0.7, 0.34, 0.34]),  # ga
                     np.array([0.34, 0.34, 0.7]),  # gi
                     np.array([0.34, 0.7, 0.34]),  # gu
                     ])


def colour_palette_vowels():
    return np.array([np.array([1, 0.3712, 0.34]),  # a
                     np.array([0.34, 0.34, 1]),  # i
                     np.array([0.34, 1, 0.34]),  # u
                     ])


def markers():
    return np.array([
        '$b$',  # ba
        '$b$',  # bi
        '$b$',  # bu
        '$d$',  # da
        '$d$',  # di
        '$d$',  # du
        '$g$',  # ga
        '$g$',  # gi
        '$g$',  # gu
    ])


def scatter_syllable(x, syllable_indices, title, dir=None, file=None, show=True, n=100):
    """
    creates scatter plot for t-SNE visualization
    :param x: 2-D latent space as output by t-SNE
    :param syllable_indices: labels for each datapoint in x, used to assign different colors to them
    :param title: title of the plot
    :param dir: directory to save the plot in
    :param file: file name to save the plot
    :param show: whether to show the plot or not
    """
    # We choose a color palette with seaborn.
    palette = colour_palette()
    marks = markers()

    # We create a scatter plot.
    plt.figure(figsize=(6, 6))  # was 8, 8, i havent tested yet
    ax = plt.subplot(aspect="equal")

    # for each loop created by chat gpt
    for i, syllable_idx in enumerate(np.unique(syllable_indices)):
        indices = np.where(syllable_indices == syllable_idx)[0]
        if len(indices) > n:
            indices = np.random.choice(indices, size=n, replace=False)
        color = np.tile(palette[i], (len(indices), 1))
        m = marks[i]
        ax.scatter(x[indices, 0], x[indices, 1],
                   lw=0,
                   s=40,
                   color=color,
                   marker=m,
                   label=translate_number_to_syllable(syllable_idx))

    plt.legend()

    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis("off")
    ax.axis("tight")

    plt.title(title)

    if file is not None:
        create_log_dir(dir)
        plt.savefig(f"{dir}/{file}", dpi=120)
        try:
            tikzplotlib.save(f"{dir}/{file}.tex")
        except:
            pass

    if show:
        plt.show()

    plt.clf()
    plt.cla()


def histogram(sequence, title, dir=None, file=None, show=True):
    # aided by ChatGPT
    # Set up the figure and color palette
    sns.set_style('whitegrid')
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = sns.color_palette('bright', n_colors=1)

    # Plot the histogram
    ax.hist(sequence, bins=100,
            color=colors[0], alpha=0.8, density=True, edgecolor='k', linewidth=1.2)

    # Compute the PDF of a standard normal distribution
    x = np.linspace(-4, 4, 1000)
    pdf = np.exp(-0.5 * x ** 2) / np.sqrt(2 * np.pi)

    # Plot the standard normal PDF
    ax.plot(x, pdf, color='r', linewidth=2)

    # Set axis labels and title
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    ax.set_title(title)

    # y axis limit
    ax.set_ylim(0, 5)

    # Save or show the plot
    if file is not None:
        create_log_dir(dir)
        plt.savefig(f"{dir}/{file}", dpi=120)
        try:
            tikzplotlib.save(f"{dir}/{file}.tex")
        except:
            pass

    if show:
        plt.show()

    plt.close(fig)


def save_audio(audio, dir, file, sample_rate=16000):
    create_log_dir(dir)
    sf.write(f"{dir}/{file}.wav", audio, sample_rate)


def resample(audio, curr_samplerate=22050, new_samplerate=16000):
    audio = torchaudio.functional.resample(
        audio, orig_freq=curr_samplerate, new_freq=new_samplerate)
    return audio


def translate_syllable_to_number(syllable):
    # syllable can be the following: ba, bi, bu, da, di, du, ga, gi, gu
    syllable_to_number = {"ba": 0, "bi": 1, "bu": 2,
                          "da": 3, "di": 4, "du": 5, "ga": 6, "gi": 7, "gu": 8}
    return syllable_to_number[syllable]


def translate_number_to_syllable(index):
    # syllable can be the following: ba, bi, bu, da, di, du, ga, gi, gu
    number_to_syllable = {0: "ba", 1: "bi", 2: "bu",
                          3: "da", 4: "di", 5: "du", 6: "ga", 7: "gi", 8: "gu"}
    return number_to_syllable[index]


def translate_syllable_vowel_number(syllable):
    # if includes "a" then 0, if includes "i" then 1, if includes "u" then 2
    return 0 if "a" in syllable else 1 if "i" in syllable else 2


def translate_vowel_number_to_vowel(number):
    # if includes "a" then 0, if includes "i" then 1, if includes "u" then 2
    return "a" if number == 0 else "i" if number == 1 else "u"


def translate_stl_number_to_class_label(number):  # stl-10 dataset
    number_to_class_label = {0: "airplane", 1: "bird", 2: "car",
                             3: "cat", 4: "deer", 5: "dog", 6: "horse", 7: "monkey", 8: "ship", 9: "truck"}
    return number_to_class_label[number]


def translate_awa2_number_to_class_label(number):  # animals with attributes dataset
    number_to_class_label = {0: "antelope", 1: "grizzly+bear", 2: "killer+whale", 3: "beaver", 4: "dalmatian",
                             5: "persian+cat", 6: "horse", 7: "german+shepherd", 8: "blue+whale", 9: "siamese+cat",
                             10: "skunk", 11: "mole", 12: "tiger", 13: "hippopotamus", 14: "leopard", 15: "moose",
                             16: "spider+monkey", 17: "humpback+whale", 18: "elephant", 19: "gorilla", 20: "ox",
                             21: "fox", 22: "sheep", 23: "seal", 24: "chimpanzee", 25: "hamster", 26: "squirrel",
                             27: "rhinoceros", 28: "rabbit", 29: "bat", 30: "giraffe", 31: "wolf", 32: "chihuahua",
                             33: "rat", 34: "weasel", 35: "otter", 36: "buffalo", 37: "zebra", 38: "giant+panda",
                             39: "deer", 40: "bobcat", 41: "pig", 42: "lion", 43: "mouse", 44: "polar+bear",
                             45: "collie", 46: "walrus", 47: "raccoon", 48: "cow", 49: "dolphin"}
    return number_to_class_label[number]


def translate_shapes3d_number_to_class_label(number):  # 3dshapes dataset
    # shapes 3d: shape (4 options)
    number_to_class_label = {0: "cube", 1: "cylinder", 2: "sphere", 3: "torus"}
    return number_to_class_label[number]


if __name__ == "__main__":
    pass
