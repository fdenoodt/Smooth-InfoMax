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

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LogHandler():
    def __init__(self, opt, logs, train_loader, criterion, gim_encoder: GIM_Encoder) -> None:
        self.opt = opt
        self.total_step = len(train_loader)
        self.logs: logger.Logger = logs
        self.criterion = criterion
        self.logging_path = f"{opt['log_path']}/{criterion.name}/GIM_L{gim_encoder.layer_depth}"
        create_log_dir(self.logging_path)


    def __call__(self, model, epoch, optimizer, train_loss, val_loss) -> None:
        self.save_train_losses(train_loss, val_loss)
        self.save_model(model, epoch, optimizer)
        self.draw_loss_curve(train_loss, val_loss)

    def save_train_losses(self, train_loss, val_loss):
        np.savetxt(f"{self.logging_path}/training_loss.csv", train_loss, delimiter=",")
        np.savetxt(f"{self.logging_path}/validation_loss.csv", val_loss, delimiter=",")
     
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
    def __init__(self, train_loader) -> None:
        self.starttime = time.time()

        self.print_idx = 100
        self.step = 0
        self.total_step = len(train_loader)

    def __call__(self, step, epoch) -> Any:
        if step % self.print_idx == 0:
            print(
                "Epoch [{}/{}], Step [{}/{}], Time (s): {:.1f}".format(
                    epoch + 1,
                    opt["num_epochs"] + opt["start_epoch"],
                    step,
                    self.total_step,
                    time.time() - self.starttime,
                )
            )

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



def show_line_sequence(sequence):
    plt.plot(sequence.to('cpu').detach().numpy())
    plt.show()


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
        enc_audios = encoder(batch_audio_signals).to(device)  # (batch_size, 512, 256)

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


# def plot_fft():
#     wave = audios[0][0].to('cpu').numpy()
#     X = np.fft.fft(wave)
#     X_mag = np.absolute(X)
#     plt.figure(figsize=(18, 10))
#     plt.plot(X_mag)  # magnitude spectrum
#     plt.xlabel('Frequency (Hz)')


