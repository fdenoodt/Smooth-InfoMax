# example python call:
# python -m interpretability.t_sne  final_bart/bart_full_audio_distribs_distr=true_kld=0 sim_audio_distr_false

import random
from typing import Optional

from sklearn.manifold import TSNE
from utils.helper_functions import *
from options_anal_hidd_repr import OPTIONS as OPT_ANAL
from arg_parser import arg_parser
from data import get_dataloader

import torch
import torch.nn as nn
import time
import numpy as np

## own modules
from config_code.config_classes import OptionsConfig, ModelType, Dataset, DataSetConfig
from models.full_model import FullModel
from options import get_options
from data import get_dataloader
from utils import logger
from arg_parser import arg_parser
from models import load_audio_model


def plot_tsne(opt: OptionsConfig, feature_space: np.ndarray, label_indices: np.ndarray, gim_name: str,
              lr: Optional[float] = 200, n_iter: Optional[int] = 1000, perplexity: Optional[int] = None):
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


def plot_histograms(feature_space_per_channel, gim_name, target_dir):
    max_dim = 16
    for idx, feature_space in enumerate(feature_space_per_channel):
        if idx == max_dim:
            break

        file = f"_ distribution_latent_space_{gim_name}_dim={idx}"

        histogram(feature_space,
                  title=f"Distributions of latent points for dimension {idx + 1} - {gim_name}", dir=target_dir,
                  file=file, show=False)

        print(f"Saved t-SNE plot to {target_dir}/{file}.png")


def main():
    opt: OptionsConfig = get_options()

    data_config = DataSetConfig(
        dataset=Dataset.DE_BOER,
        split_in_syllables=True,
        batch_size=128,
        limit_train_batches=1.0,
        limit_validation_batches=1.0,
        labels="syllables"
    )

    classifier_config = opt.syllables_classifier_config

    arg_parser.create_log_path(opt, add_path_var="post_hoc")

    # random seeds
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    existing_model = False
    context_model, _ = load_audio_model.load_model_and_optimizer(
        opt,
        classifier_config,
        reload_model=existing_model,
        calc_accuracy=True,
        num_GPU=1,
    )
    context_model.eval()

    train_loader, _, test_loader, _ = get_dataloader.get_dataloader(data_config)
    logs = logger.Logger(opt)

    # gather the data
    all_audio = np.array([])
    all_labels = np.array([])
    nb_channels = context_model.module.output_dim
    for i, (audio, _, label, _) in enumerate(train_loader):
        encoder: FullModel = context_model.module
        audio = audio.to(opt.device)

        with torch.no_grad():
            audio = encoder.forward_through_all_modules(audio)
            audio = audio.cpu().detach().numpy()  # (batch_size, seq_len, nb_channels)

            # vstack the audio
            if all_audio.size == 0:
                all_audio = audio
                all_labels = label
            else:
                all_audio = np.vstack((all_audio, audio))
                all_labels = np.hstack((all_labels, label))

    # (batch_size, seq_len, nb_channels) -> (batch_size, seq_len * nb_channels)
    all_audio = all_audio.reshape(all_audio.shape[0], -1)
    print(all_audio.shape)
    print(all_labels.shape)

    n = all_labels.shape[0]  # sqrt(1920) ~= 44

    # t-SNE
    params = [('auto', 1000, float(np.sqrt(n))),
              (200, 1000, 50), (200, 1000, 30), (200, 1000, 10),
              (20, 1000, 50), (20, 1000, 30), (20, 1000, 10),
              (1000, 1000, 50), (1000, 1000, 30), (1000, 1000, 10)]
    # plot_tsne(opt, all_audio, all_labels, "SIM", lr=200, n_iter=1000, perplexity=50)
    for lr, n_iter, perplexity in params:
        plot_tsne(opt, all_audio, all_labels, f"ALL_SIM_{lr}_{n_iter}_{perplexity}",
                  lr=lr, n_iter=n_iter, perplexity=perplexity)


    # reshape to original shape
    all_audio = all_audio.reshape(all_audio.shape[0], -1, nb_channels)  # (batch_size, seq_len, nb_channels)

    # mean of seq len
    all_audio = np.mean(all_audio, axis=1)  # (batch_size, nb_channels)
    params = [('auto', 1000, float(np.sqrt(n))),
              (200, 1000, 50), (200, 1000, 30), (200, 1000, 10),
              (20, 1000, 50), (20, 1000, 30), (20, 1000, 10),
              (1000, 1000, 50), (1000, 1000, 30), (1000, 1000, 10)]
    # plot_tsne(opt, all_audio, all_labels, "SIM", lr=200, n_iter=1000, perplexity=50)
    for lr, n_iter, perplexity in params:
        plot_tsne(opt, all_audio, all_labels, f"MEAN_SIM_{lr}_{n_iter}_{perplexity}",
                  lr=lr, n_iter=n_iter, perplexity=perplexity)



if __name__ == "__main__":
    main()
    print("Finished")
