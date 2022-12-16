# %%
# import torch
# import numpy as np
# from models import load_audio_model
import os
import IPython.display as ipd
from options import OPTIONS as opt
# from models import full_model
from utils import model_utils
# import torch
# import torch.nn as nn
# import os
# from data import get_dataloader

import torch
import time
import numpy as np
import random
import gc
from options import OPTIONS

# own modules
from utils import logger
from arg_parser import arg_parser
from models import load_audio_model
from data import get_dataloader
from validation import val_by_latent_speakers
from validation import val_by_InfoNCELoss
from models import full_model

# %%

import os
import matplotlib.pyplot as plt
import librosa
import librosa.display
import IPython.display as ipd


def plot_spectrogram(signal, name):
    """Compute power spectrogram with Short-Time Fourier Transform and plot result."""
    spectrogram = librosa.amplitude_to_db(librosa.stft(signal))
    plt.figure(figsize=(20, 15))
    librosa.display.specshow(spectrogram, y_axis="log")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Log-frequency power spectrogram for {name}")
    plt.xlabel("Time")
    plt.show()


# %%
def load_model(path):
    # Code comes from: def load_model_and_optimizer()
    kernel_sizes = [10, 8, 4, 4, 4]
    strides = [5, 4, 2, 2, 2]
    padding = [2, 2, 2, 2, 1]
    enc_hidden = 512
    reg_hidden = 256

    calc_accuracy = False
    reload_model = True
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
    model.load_state_dict(torch.load(path))

    return model, optimizer

# %%


if __name__ == "__main__":
    arg_parser.create_log_path(opt)

    model, _ = load_model(path='./g_drive_model/model_180.ckpt')
    model.eval()

    logs = logger.Logger(opt)

    train_loader, train_dataset, test_loader, test_dataset = get_dataloader.get_de_boer_sounds_data_loaders(
        opt
    )
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # %%

    audios = None
    filenames = None
    start_idxs = None
    for step, (audio, filename, _, start_idx) in enumerate(train_loader):
        audios = audio
        filenames = filename
        start_idxs = start_idx
        break

    audios.shape

    # %%
    ipd.Audio(os.path.join(
        r"C:\GitHub\thesis-fabian-denoodt\GIM\datasets\gigabo\train", f"{filenames[0]}.wav"), rate=44100)

    # %%
    audios[0].shape
    ipd.Audio(audios[0], rate=44100)

    # # %%
    # plot_spectrogram(audios[0].to('cpu').numpy()[0], "bibaga")

    # %%

    model(audios).shape

    # %%

    model.module.fullmodel

    # %%

    # model_input = audio.to(opt["device"])
    model_input = audios.to('cuda')
    big_feature_space = []

    for idx, layer in enumerate(model.module.fullmodel):
        context, z = layer.get_latents(model_input)
        model_input = z.permute(0, 2, 1)

        print(z.shape)
        # model_input = z.permute(0, 2, 1)
        # latent_rep = context.permute(0, 2, 1).cpu().numpy()



    # %%

    wave = audios[0][0].to('cpu').numpy()
    import numpy as np
    X = np.fft.fft(wave)
    X_mag = np.absolute(X)
    f = np.linspace(0, _, len(X_mag))

    plt.figure(figsize=(18, 10))
    plt.plot(f, X_mag) # magnitude spectrum
    plt.xlabel('Frequency (Hz)')