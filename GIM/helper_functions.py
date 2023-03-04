# %%
import soundfile as sf
import IPython.display as ipd
from GIM_encoder import GIM_Encoder
from options import OPTIONS as opt
import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd
from utils import logger
import os
from typing import Any
import numpy as np
import time
from torchvision import transforms
import torchaudio

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LogHandler():
    '''
    This class handles the logging of the training process.
    '''
    def __init__(self, opt, logs, train_loader, criterion, gim_encoder: GIM_Encoder, learning_rate) -> None:
        self.opt = opt
        self.total_step = len(train_loader)
        self.logs: logger.Logger = logs
        self.criterion = criterion
        self.logging_path = f"{opt['log_path']}/{criterion.name}/lr_{learning_rate:.7f}/GIM_L{gim_encoder.layer_depth}"
        create_log_dir(self.logging_path)

    def __call__(self, model, epoch, optimizer, train_loss, val_loss) -> None:
        self.save_train_losses(train_loss, val_loss)
        self.save_model(model, epoch, optimizer)
        self.draw_loss_curve(train_loss, val_loss)

    def save_train_losses(self, train_loss, val_loss):
        np.savetxt(f"{self.logging_path}/training_loss.csv",
                   train_loss, delimiter=",")
        np.savetxt(f"{self.logging_path}/validation_loss.csv",
                   val_loss, delimiter=",")

    def save_model(self, model, epoch, optimizer) -> None:
        torch.save(model.state_dict(), f'{self.logging_path}/model_{epoch}.pt')

    def draw_loss_curve(self, train_loss, val_loss):
        assert len(train_loss) == len(val_loss)

        lst_iter = np.arange(len(train_loss))
        plt.plot(lst_iter, np.array(train_loss), "-b", label="train loss")

        lst_iter = np.arange(len(val_loss))
        plt.plot(lst_iter, np.array(val_loss), "-r", label="val loss")

        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend(loc="upper right")

        # save image
        plt.savefig(os.path.join(self.logging_path, "loss.png"))
        plt.close()


class EpochPrinter():
    def __init__(self, train_loader, learning_rate, criterion, decoder_depth) -> None:
        self.starttime = time.time()

        self.print_idx = 100
        self.step = 0
        self.total_step = len(train_loader)
        self.learning_rate = learning_rate
        self.criterion = criterion
        self.decoder_depth = decoder_depth

    def __call__(self, step, epoch) -> Any:
        if step % self.print_idx == 0:
            print(f"Epoch[{epoch + 1}/{opt['num_epochs'] + opt['start_epoch']}], Step[{step}/{self.total_step, }], Time(s): {time.time() - self.starttime: .1f} L: {self.decoder_depth} lr: {self.learning_rate}, {self.criterion.name}")


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
        enc_audios = encoder(batch_audio_signals).to(
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
    #detach + numpy
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
    return x_mag

# if 16khz, only 8000 frequencies possible -> sample rate should be twice as large as the highest frequency


def plot_two_graphs_side_by_side(sequence1, sequence2, title="True vs Predicted", dir=None, file=None, show=True):
    ''' Plot two graphs side by side '''
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(title)

    ax1.plot(sequence1)
    ax2.plot(sequence2)

    if file is not None:
        create_log_dir(dir)
        plt.savefig(f"{dir}/{file}")

    if show:
        plt.show()

    plt.clf()

def plot_four_graphs_side_by_side(sequence1, sequence2, sequence3, sequence4, title="True vs Predicted", dir=None, file=None, show=True):
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
