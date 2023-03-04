# %%
"""
This file is used to analyse the hidden representation of the audio signal.
- It stores the hidden representation of the audio signal for each batch in a tensor.
- The tensor is then visualised using a scatter plot.
"""
import importlib
import random
import numpy as np
import torch
from GIM_encoder import GIM_Encoder
import helper_functions
from options import OPTIONS as opt

from arg_parser import arg_parser
from data import get_dataloader


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
    # plt.show()


def _save_encodings(root_dir, data_type, encoder: GIM_Encoder, data_loader):
    assert data_type in ["train", "test"]

    # audio, filename, pronounced_syllable, full_word
    for idx, (batch_org_audio, filenames, _, _) in enumerate(iter(data_loader)):
        batch_org_audio = batch_org_audio.to(device)
        batch_enc_audio_per_module = encoder(batch_org_audio)

        for module_idx, batch_enc_audio in enumerate(batch_enc_audio_per_module):
            target_dir = f"{root_dir}/module={module_idx + 1}/{data_type}/" # eg: 01GIM_L{layer_depth}/module=1/train/
            create_log_dir(target_dir)

            print(f"Batch {idx} - {batch_enc_audio.shape} - Mean: {torch.mean(batch_enc_audio)} - Std: {torch.std(batch_enc_audio)}")

            torch.save(batch_enc_audio, f"{target_dir}/batch_encodings_{idx}.pt")
            torch.save(filenames, f"{target_dir}/batch_filenames_{idx}.pt")


def generate_and_save_encodings(encoder, train_loader, test_loader, split: bool):
    target_dir = f"{LOG_PATH}/hidden_repr/{'split' if split else 'full'}"

    _save_encodings(target_dir, "train", encoder, train_loader)
    _save_encodings(target_dir, "test", encoder, test_loader)
    

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
                break
        break

def generate_visualisations(GIM_model_name):
    saved_files_dir = f"{LOG_PATH}/hidden_repr/{GIM_model_name}/"
    train_dir = f"{saved_files_dir}/train"
    test_dir = f"{saved_files_dir}/test/"

    target_dir = f"{LOG_PATH}/hidden_repr_vis/{GIM_model_name}/"
    train_vis_dir = f"{target_dir}/train"
    test_vis_dir = f"{target_dir}/test/"
    create_log_dir(train_vis_dir)
    create_log_dir(test_vis_dir)

    _generate_visualisations(train_dir, GIM_model_name, train_vis_dir)
    _generate_visualisations(test_dir, GIM_model_name, test_vis_dir)


# old model that was trained on larger samples:
# DRIVE LOGS/03 MODEL noise 400 epochs/logs/audio_experiment/

ENCODER_MODEL_DIR = r"C:\GitHub\thesis-fabian-denoodt\GIM\logs\audio_experiment_test_w_ar"
# ENCODER_MODEL_DIR = r"C:\GitHub\thesis-fabian-denoodt\GIM\logs\audio_experiment_3_lr_noise"
LOG_PATH = f"{ENCODER_MODEL_DIR}/analyse_hidden_repr/"
EPOCH_VERSION = 1
AUTO_REGRESSOR_AFTER_MODULE = True

if __name__ == "__main__":
    torch.cuda.empty_cache()
    arg_parser.create_log_path(opt)
    opt['batch_size'] = 64 + 32
    opt['batch_size_multiGPU'] = opt['batch_size']
    opt['auto_regressor_after_module'] = AUTO_REGRESSOR_AFTER_MODULE

    logs = logger.Logger(opt)

    ENCODER_NAME = f"model_{EPOCH_VERSION}.ckpt"
    ENCODER_MODEL_PATH = f"{ENCODER_MODEL_DIR}/{ENCODER_NAME}"
    
    # model consisting of single module
    # experiment_name = 'RMSE_decoder_GIM_layer3_MSE_SPECTRAL_loss'
    # opt['experiment'] = experiment_name
    # opt['save_dir'] = f'{experiment_name}_experiment'
    # opt['log_path'] = f'./logs/{experiment_name}_experiment'
    # opt['log_path_latent'] = f'./logs/{experiment_name}_experiment/latent_space'
    # opt['batch_size_multiGPU'] = opt['batch_size']
    # opt['num_epochs'] = 50



    # **** Full audio samples ****
    # load the data: full audio samples
    split = True
    train_loader, _, test_loader, _ = get_dataloader.get_de_boer_sounds_data_loaders(opt, shuffle=False, split_and_pad=split, train_noise=False)

    ENCODER: GIM_Encoder = GIM_Encoder(opt, path=ENCODER_MODEL_PATH)
    generate_and_save_encodings(ENCODER, train_loader, test_loader, split)
    # generate_visualisations(gim_name)



    # **** audio samples on syllables ****

    torch.cuda.empty_cache()

