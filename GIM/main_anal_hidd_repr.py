# %%
import importlib
from GIM_encoder import GIM_Encoder
import helper_functions
from options import OPTIONS as opt
import torch

from arg_parser import arg_parser
from data import get_dataloader
import numpy as np
import random

random.seed(0)

if(True):
    importlib.reload(helper_functions)
    from helper_functions import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == "__main__":
    torch.cuda.empty_cache()
    arg_parser.create_log_path(opt)

    experiment_name = 'RMSE_decoder_GIM_layer3_MSE_SPECTRAL_loss'
    opt['experiment'] = experiment_name
    opt['save_dir'] = f'{experiment_name}_experiment'
    opt['log_path'] = f'./logs/{experiment_name}_experiment'
    opt['log_path_latent'] = f'./logs/{experiment_name}_experiment/latent_space'
    opt['num_epochs'] = 50
    opt['batch_size'] = 64 + 32
    opt['batch_size_multiGPU'] = opt['batch_size']

    create_log_dir(opt['log_path'])

    logs = logger.Logger(opt)

    # load the data
    train_loader, _, test_loader, _ = get_dataloader.\
        get_de_boer_sounds_decoder_data_loaders(opt)

    encoder = GIM_Encoder(
        opt, layer_depth=3, path="DRIVE LOGS/03 MODEL noise 400 epochs/logs/audio_experiment/model_360.ckpt")

    (org_audio, _, _, _) = next(iter(test_loader))
    org_audio = org_audio.to(device)
    enc_audio = encoder(org_audio).to(device)

    # torch.cuda.empty_cache()

    # %%
    log_path = "analyse_hidden_repr/hidden_repr_vis/01GIM_L3"
    import cv2
    import matplotlib.cm as cm

    def visualise_3d_tensor(tensor, name):
        nd_arr = tensor.to('cpu').numpy()
        nb_channels, length = nd_arr.shape

        nd_arr_flat = nd_arr.flatten()  # (nb_channels * length)
        s = nd_arr_flat / np.max(nd_arr_flat)
        xs = np.repeat(np.arange(0, nb_channels, 1), length)  # channels
        ys = np.tile(np.arange(0, length, 1), nb_channels)  # length

        fig, ax = plt.subplots(figsize=(20, 20))
        ax.scatter(ys, xs, s=1000*(s**4), marker="s", c='orange', alpha=0.3)
        ax.set_aspect('equal')
        ax.set_xlabel('Signal length')
        ax.set_ylabel('Channels')
        ax.set_title('Hidden representation of the audio signal - Layer 3')

        # Show the plot
        plt.savefig(f"{log_path}/{name}.png")
        plt.show()

    # tensor = torch.zeros(3, 256, 256)
    # tensor[0, :, :] = 1
    # enc_audio.shape = (96, 512, 256)
    visualise_3d_tensor(enc_audio[0], "test")

    # %%
    import matplotlib.pyplot as plt
    import numpy as np

    # Generate a 2D numpy array
    arr = np.random.rand(10, 2)

    # Plot the array as a scatter plot
    plt.scatter(arr[:, 0], arr[:, 1], s=arr[:, 1]*100)

    # Add labels and title to the plot
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.title('Scatter plot with size based on value in the 2D array')

    # Show the plot
    plt.show()

# %%
