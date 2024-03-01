# example python call:
# python -m post_hoc_analysis.interpretability.t_sne  final_bart/bart_full_audio_distribs_distr=true_kld=0 sim_audio_distr_false

import random
from typing import Optional, Union

from sklearn.manifold import TSNE
from utils.helper_functions import *
from options_anal_hidd_repr import OPTIONS as OPT_ANAL
from arg_parser import arg_parser
from data import get_dataloader

import torch
import torch.nn as nn
import time
import numpy as np

import os
import wandb

## own modules
from config_code.config_classes import OptionsConfig, ModelType, Dataset, DataSetConfig
from models.full_model import FullModel
from options import get_options
from data import get_dataloader
from utils import logger
from arg_parser import arg_parser
from models import load_audio_model


def plot_tsne(opt: OptionsConfig, feature_space: np.ndarray, label_indices: np.ndarray, gim_name: str,
              lr: Union[float, str], n_iter: int, perplexity: int, wandb_is_on: bool):
    # eg target_dir = 'analyse_hidden_repr//hidden_repr_vis/split/module=1/test/'

    projection = TSNE(
        init='random',
        learning_rate=lr,
        n_iter=n_iter,
        perplexity=perplexity
        # n_iter_without_progress=1000,
        # perplexity=50
        # perplexity=15
    ).fit_transform(feature_space)

    assert projection.shape[0] == label_indices.shape[0]

    # file = f"_ t-SNE_latent_space_{gim_name}"
    file = f"_ t-SNE_latent_space_{gim_name}.png"  # Add the file extension here

    save_dir = opt.log_path

    scatter(projection, label_indices,
            title=f"t-SNE Latent space - {gim_name}", dir=save_dir, file=file, show=False)

    print(f"Saved t-SNE plot to {save_dir}/{file}.png")

    if wandb_is_on:
        wandb.log({f"t-SNE_latent_space_{gim_name}": [wandb.Image(f"{save_dir}/{file}.png")]})


def plot_histograms(opt: OptionsConfig, feature_space_per_channel, gim_name, max_dim: int, wandb_is_on: bool):
    # feature_space_per_channel: (nb_channels, batch_size, seq_len)
    save_dir = opt.log_path

    for idx, feature_space in enumerate(feature_space_per_channel):
        # feature_space: (batch_size, seq_len)
        if idx == max_dim:
            break

        feature_space = feature_space.flatten()
        file = f"_ distribution_latent_space_{gim_name}_dim={idx}"

        histogram(feature_space,
                  title=f"Distributions of latent points for dimension {idx + 1} - {gim_name}", dir=save_dir,
                  file=file, show=False)

        print(f"Saved t-SNE plot to {save_dir}/{file}.png")

        if wandb_is_on and idx < 32:  # max_images = 32 images
            wandb.log({f"distribution_latent_space_{gim_name}_dim={idx}": [wandb.Image(f"{save_dir}/{file}.png")]})


def _get_data_from_loader(loader, encoder: FullModel, opt: OptionsConfig, final_module: str):
    assert final_module in ["final", "final_cnn"]

    all_audio = np.array([])
    all_labels = np.array([])

    for i, (audio, _, label, _) in enumerate(loader):
        audio = audio.to(opt.device)

        with torch.no_grad():
            if final_module == "final":
                audio = encoder.forward_through_all_modules(audio)
            else:
                audio = encoder.forward_through_all_cnn_modules(audio)  # only cnn modules have kl divergence
            audio = audio.cpu().detach().numpy()  # (batch_size, seq_len, nb_channels)

            # vstack the audio
            if all_audio.size == 0:
                all_audio = audio
                all_labels = label
            else:
                all_audio = np.vstack((all_audio, audio))
                all_labels = np.hstack((all_labels, label))

    # If output from final_cnn, permute channels and seq_len
    if final_module == "final_cnn":
        all_audio = np.moveaxis(all_audio, 2, 1)

    return all_audio, all_labels


def main():
    opt: OptionsConfig = get_options()

    classifier_config = opt.syllables_classifier_config

    # Check if the wandb_run_id.txt file exists
    wandb_is_on = False
    if os.path.exists(os.path.join(opt.log_path, 'wandb_run_id.txt')):
        # If the file exists, read the run id from the file
        with open(os.path.join(opt.log_path, 'wandb_run_id.txt'), 'r') as f:
            run_id = f.read().strip()

        # Initialize a wandb run with the same run id
        wandb.init(id=run_id, resume="allow")
        wandb_is_on = True

    arg_parser.create_log_path(opt, add_path_var="post_hoc")

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    load_existing_model = True
    context_model, _ = load_audio_model.load_model_and_optimizer(
        opt,
        classifier_config,
        reload_model=load_existing_model,
        calc_accuracy=True,
        num_GPU=1,
    )
    context_model.eval()
    logs = logger.Logger(opt)
    nb_channels = context_model.module.output_dim

    data_config = DataSetConfig(
        dataset=Dataset.DE_BOER,
        split_in_syllables=True,
        batch_size=128,
        limit_train_batches=1.0,
        limit_validation_batches=1.0,
        labels="syllables"
    )

    # retrieve data for classifier
    train_loader_syllables, _, test_loader_syllables, _ = get_dataloader.get_dataloader(data_config)
    all_audio, all_labels = _get_data_from_loader(train_loader_syllables, context_model.module, opt, "final")
    n = all_labels.shape[0]  # sqrt(1920) ~= 44

    # mean of seq len
    all_audio_mean = np.mean(all_audio, axis=1)  # (batch_size, nb_channels)
    lr, n_iter, perplexity = ('auto', 1000, int(float(np.sqrt(n))))
    plot_tsne(opt, all_audio_mean, all_labels, f"MEAN_SIM_{lr}_{n_iter}_{perplexity}",
              lr=lr, n_iter=n_iter, perplexity=perplexity, wandb_is_on=wandb_is_on)

    # retrieve full data that encoder was trained on
    data_config.split_in_syllables = False
    train_loader_full, _, test_loader, _ = get_dataloader.get_dataloader(data_config)
    all_audio, all_labels = _get_data_from_loader(train_loader_full, context_model.module, opt, "final_cnn")

    # plot histograms
    # (batch_size, seq_len, nb_channels) -> (nb_channels, batch_size, seq_len)
    audio_per_channel = np.moveaxis(all_audio, 2, 0)
    plot_histograms(opt, audio_per_channel, f"MEAN_SIM", max_dim=32, wandb_is_on=wandb_is_on)

    print("Finished")
    if wandb_is_on:
        wandb.finish()


if __name__ == "__main__":
    main()
    print("Finished")
