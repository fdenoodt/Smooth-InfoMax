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
import cv2
import matplotlib.cm as cm

log_path = "analyse_hidden_repr/"

random.seed(0)

if(True):
    importlib.reload(helper_functions)
    from helper_functions import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def visualise_2d_tensor(tensor, GIM_model_name, target_dir, name):
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
    ax.set_title(f'Hidden representation of the audio signal - {GIM_model_name} - {name}')

    # Show the plot
    plt.savefig(f"{target_dir}/{name}.png")
    plt.show()


def _save_encodings(target_dir, encoder, data_loader):
    for idx, (batch_org_audio, filenames, _, _) in enumerate(iter(data_loader)):
        batch_org_audio = batch_org_audio.to(device)
        batch_enc_audio = encoder(batch_org_audio)

        torch.save(batch_enc_audio, f"{target_dir}/batch_encodings_{idx}.pt")
        torch.save(filenames, f"{target_dir}/batch_filenames_{idx}.pt")


def generate_and_save_encodings(encoder, train_loader, test_loader, GIM_model_name):
    target_dir = f"{log_path}/hidden_repr/{GIM_model_name}/"
    train_dir = f"{target_dir}/train"
    test_dir = f"{target_dir}/test/"
    create_log_dir(train_dir)
    create_log_dir(test_dir)

    _save_encodings(train_dir, encoder, train_loader)
    _save_encodings(test_dir, encoder, test_loader)
    
    # visualise_2d_tensor(enc_audio[0], "01GIM_L3", f"{name}_encodings")

def _generate_visualisations(data_dir, GIM_model_name, target_dir):
    # iterate over files in train_dir
    for file in os.listdir(data_dir): # Generated via copilot
        if file.endswith(".pt") and file.startswith("batch_encodings"):
            # load the file
            batch_encodings = torch.load(f"{data_dir}/{file}")
            batch_filenames = torch.load(f"{data_dir}/{file.replace('encodings', 'filenames')}")
            for idx, (enc, name) in enumerate(zip(batch_encodings, batch_filenames)):
                name = name.split("_")[0] # eg: babugu_1 -> babugu
                visualise_2d_tensor(enc, GIM_model_name, target_dir, f"{name}")

def generate_visualisations(GIM_model_name):
    saved_files_dir = f"{log_path}/hidden_repr/{GIM_model_name}/"
    train_dir = f"{saved_files_dir}/train"
    test_dir = f"{saved_files_dir}/test/"

    target_dir = f"{log_path}/hidden_repr_vis/{GIM_model_name}/"
    train_vis_dir = f"{target_dir}/train"
    test_vis_dir = f"{target_dir}/test/"
    create_log_dir(train_vis_dir)
    create_log_dir(test_vis_dir)

    _generate_visualisations(train_dir, GIM_model_name, train_vis_dir)
    _generate_visualisations(test_dir, GIM_model_name, test_vis_dir)


                
                # {log_path}/hidden_repr_vis/


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

    logs = logger.Logger(opt)

    # load the data
    # train_loader, _, test_loader, _ = get_dataloader.\
    #     get_de_boer_sounds_decoder_data_loaders(opt)

    # layer_depth = 3
    # encoder = GIM_Encoder(opt, layer_depth=layer_depth, path="DRIVE LOGS/03 MODEL noise 400 epochs/logs/audio_experiment/model_360.ckpt")

    # generate_and_save_encodings(encoder, train_loader, test_loader, "01GIM_L3")
    generate_visualisations("01GIM_L3")



    torch.cuda.empty_cache()

    # %%
    # visualise_3d_tensor(enc_audio[0], "test")
    # %%
