import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
import torch
import wandb
from sklearn.manifold import TSNE

from arg_parser import arg_parser
## own modules
from config_code.config_classes import OptionsConfig, Loss
from data import get_dataloader
from options import get_options
from utils import logger
from utils.helper_functions import create_log_dir
from utils.utils import set_seed
from vision.models import load_vision_model
from vision.models.FullModel import FullVisionModel


def _get_data_from_loader(train_loader: torch.utils.data.DataLoader,
                          context_model: FullVisionModel, opt: OptionsConfig, num_datapoints: Optional[int] = None):
    all_data = np.array([])
    all_labels = np.array([])
    datapoints_processed = 0

    for i, (data, label) in enumerate(train_loader):
        data = data.to(opt.device)

        with torch.no_grad():
            _, _, _, _, data, _ = context_model(data, label)  # no clue why also label is passed here
            data = data.cpu().detach().numpy()  # (batch_size, seq_len, nb_channels)

            # vstack the data
            if all_data.size == 0:
                all_data = data
                all_labels = label
            else:
                all_data = np.vstack((all_data, data))
                all_labels = np.hstack((all_labels, label))

            datapoints_processed += data.shape[0]

            # If num_datapoints is not None and we have processed enough datapoints, break the loop
            if num_datapoints is not None and datapoints_processed >= num_datapoints:
                break

    return all_data, all_labels


def scatter_generic(x, labels, title, dir=None, file=None, show=True, n=100):
    """
    creates scatter plot for t-SNE visualization
    :param x: 2-D latent space as output by t-SNE
    :param labels: labels for each datapoint in x, used to assign different colors to them
    :param title: title of the plot
    :param dir: directory to save the plot in
    :param file: file name to save the plot
    :param show: whether to show the plot or not
    """
    # We choose a color palette with seaborn.
    # palette = colour_palette()

    # We create a scatter plot.
    plt.figure(figsize=(6, 6))  # was 8, 8, i havent tested yet
    ax = plt.subplot(aspect="equal")

    # for each loop created by chat gpt
    for i, label in enumerate(np.unique(labels)):
        indices = np.where(labels == label)[0]
        if len(indices) > n:
            indices = np.random.choice(indices, size=n, replace=False)
        # color = np.tile(palette[i], (len(indices), 1))
        ax.scatter(x[indices, 0], x[indices, 1],
                   lw=0,
                   s=40,
                   # color=color,
                   label=label)

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


def plot_tsne_vision(opt, all_data_mean, all_labels, param, lr, n_iter, perplexity, wandb_is_on):
    # flatten the dataset
    n_samples, n_features, n_channels = all_data_mean.shape
    all_data_mean = all_data_mean.reshape((n_samples, n_features * n_channels))

    projection = TSNE(
        init='random',
        learning_rate=lr,
        n_iter=n_iter,
        perplexity=perplexity
    ).fit_transform(all_data_mean)

    assert projection.shape[0] == all_labels.shape[0]

    file = f"_ t-SNE_latent_space_{param}.png"  # Add the file extension here
    save_dir = opt.log_path

    scatter_generic(projection, all_labels,
                    title=f"t-SNE Latent space - {param}", dir=save_dir, file=file, show=False)

    print(f"Saved t-SNE plot to {save_dir}/{file}")

    if wandb_is_on:
        wandb.log({f"LatSpace/t-SNE_latent_space_{param}": [wandb.Image(f"{save_dir}/{file}")]})

    print(f"Saved t-SNE plot to {save_dir}/{file}")


def main():
    opt: OptionsConfig = get_options()
    USE_WANDB = opt.use_wandb
    opt.loss = Loss.SUPERVISED_VISUAL

    wandb_is_on = False
    if USE_WANDB:
        # Check if the wandb_run_id.txt file exists
        if os.path.exists(os.path.join(opt.log_path, 'wandb_run_id.txt')):
            # If the file exists, read the run id from the file
            with open(os.path.join(opt.log_path, 'wandb_run_id.txt'), 'r') as f:
                run_id = f.read().strip()

            # Initialize a wandb run with the same run id
            dataset = opt.vision_classifier_config.dataset.dataset
            wandb.init(project=f"SIM_VISION_ENCODER_{dataset}", id=run_id, resume="allow")
            wandb_is_on = True

    arg_parser.create_log_path(opt, add_path_var="post_hoc")

    # random seeds
    set_seed(opt.seed)

    context_model, _ = load_vision_model.load_model_and_optimizer(
        opt, reload_model=True, calc_loss=False, classifier_config=opt.vision_classifier_config
    )
    context_model.module.switch_calc_loss(False)
    context_model.eval()

    context_model.eval()
    logs = logger.Logger(opt)

    # retrieve data for classifier
    _, _, train_loader, _, test_loader, _ = get_dataloader.get_dataloader(opt.encoder_config.dataset,
                                                                          purpose_is_unsupervised_learning=False)
    all_data, all_labels = _get_data_from_loader(train_loader, context_model, opt, num_datapoints=None)
    n = all_labels.shape[0]

    # mean of seq len
    all_data_mean = np.mean(all_data, axis=1)  # (batch_size, nb_channels)
    lr, n_iter, perplexity = ('auto', 1000, int(float(np.sqrt(n))))
    plot_tsne_vision(opt, all_data_mean, all_labels, f"MEAN_SIM_{lr}_{n_iter}_{perplexity}",
                     lr=lr, n_iter=n_iter, perplexity=perplexity, wandb_is_on=wandb_is_on)

    # data_config.labels = 'vowels'
    # train_loader_syllables, _, test_loader_syllables, _ = get_dataloader.get_dataloader(data_config)
    # all_audio, all_labels = _get_data_from_loader(train_loader_syllables, context_model.module, opt, "final_cnn")
    # n = all_labels.shape[0]  # sqrt(1920) ~= 44
    #
    # _audio_per_channel = np.moveaxis(all_audio, 1, 0)
    # scatter_3d(_audio_per_channel[0], _audio_per_channel[1], _audio_per_channel[2],
    #            all_labels, title=f"3D Latent Space of the First Three Dimensions", dir=opt.log_path,
    #            file=f"_ 3D latent space idices 0_1_2", show=False, wandb_is_on=wandb_is_on)
    # #
    # # retrieve full data that encoder was trained on
    # data_config.split_in_syllables = False
    # train_loader_full, _, test_loader, _ = get_dataloader.get_dataloader(data_config)
    # all_audio, all_labels = _get_data_from_loader(train_loader_full, context_model.module, opt, "final_cnn")
    #
    # # plot histograms
    # # (batch_size, seq_len, nb_channels) -> (nb_channels, batch_size, seq_len)
    # audio_per_channel = np.moveaxis(all_audio, 2, 0)
    # plot_histograms(opt, audio_per_channel, f"MEAN_SIM", max_dim=32, wandb_is_on=wandb_is_on)
    #
    # print("Finished")
    # if wandb_is_on:
    #     wandb.finish()


if __name__ == '__main__':
    main()
