import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from sklearn.manifold import TSNE

from arg_parser import arg_parser
## own modules
from config_code.config_classes import OptionsConfig, Loss, Dataset
from options import get_options
from post_hoc_analysis.interpretability.interpretabil_util import scatter_3d_generic, plot_histograms
from utils import logger
from utils.helper_functions import create_log_dir, translate_stl_number_to_class_label, \
    translate_shapes3d_number_to_class_label, translate_awa2_number_to_class_label
from utils.utils import set_seed, retrieve_existing_wandb_run_id
from vision.data import get_dataloader
from vision.models import load_vision_model
from vision.models.FullModel import FullVisionModel

try:
    import tikzplotlib
except:
    pass


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


def scatter_generic(x, labels, title, translate_idx_to_label_fn: callable, dir=None, file=None, show=True, n=100):
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
    for i, label_idx in enumerate(np.unique(labels)):
        indices = np.where(labels == label_idx)[0]
        if len(indices) > n:
            indices = np.random.choice(indices, size=n, replace=False)
        # color = np.tile(palette[i], (len(indices), 1))
        ax.scatter(x[indices, 0], x[indices, 1],
                   lw=0,
                   s=40,
                   # color=color,
                   label=translate_idx_to_label_fn(label_idx))

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


def plot_tsne_vision(opt, all_data_mean, all_labels, param, lr, n_iter, perplexity, wandb_is_on,
                     translate_idx_to_label_fn: callable):
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
                    title=f"t-SNE Latent space - {param}",
                    translate_idx_to_label_fn=translate_idx_to_label_fn,
                    dir=save_dir, file=file, show=False)

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
        wandb_is_on = False
        run_id, project_name = retrieve_existing_wandb_run_id(opt)
        if run_id is not None:
            # Initialize a wandb run with the same run id
            wandb.init(id=run_id, resume="allow", project=project_name, entity=opt.wandb_entity)
            wandb_is_on = True

    arg_parser.create_log_path(opt, add_path_var="post_hoc")

    # random seeds
    set_seed(opt.seed)

    context_model, _ = load_vision_model.load_model_and_optimizer(
        opt, reload_model=True, calc_loss=False, downstream_config=opt.vision_classifier_config
    )
    context_model.module.switch_calc_loss(False)
    context_model.eval()

    context_model.eval()
    logs = logger.Logger(opt)

    dataset = opt.encoder_config.dataset
    if dataset.dataset == Dataset.STL10:
        translate_idx_to_label_fn: callable = translate_stl_number_to_class_label
    elif dataset.dataset in [Dataset.SHAPES_3D, Dataset.SHAPES_3D_SUBSET]:
        translate_idx_to_label_fn: callable = translate_shapes3d_number_to_class_label
    elif dataset.dataset == Dataset.ANIMAL_WITH_ATTRIBUTES:
        translate_idx_to_label_fn: callable = translate_awa2_number_to_class_label
    else:
        raise ValueError(f"Unknown dataset: {dataset.dataset}")

    ### t-SNE
    _, _, train_loader, _, test_loader, _ = get_dataloader.get_dataloader(dataset,
                                                                          purpose_is_unsupervised_learning=False)
    all_data, all_labels = _get_data_from_loader(train_loader, context_model, opt, num_datapoints=None)
    n = all_labels.shape[0]

    # mean of seq len
    all_data_mean = np.mean(all_data, axis=1)  # (batch_size, nb_channels)
    lr, n_iter, perplexity = ('auto', 1000, int(float(np.sqrt(n))))
    plot_tsne_vision(opt, all_data_mean, all_labels, f"MEAN_SIM_{lr}_{n_iter}_{perplexity}",
                     lr=lr, n_iter=n_iter, perplexity=perplexity, wandb_is_on=wandb_is_on,
                     translate_idx_to_label_fn=translate_idx_to_label_fn)

    ### 3D scatter plot
    _, _, train_loader, _, test_loader, _ = get_dataloader.get_dataloader(dataset,
                                                                          purpose_is_unsupervised_learning=False)
    all_data, all_labels = _get_data_from_loader(train_loader, context_model, opt, num_datapoints=None)
    n = all_labels.shape[0]

    _data_per_channel = np.moveaxis(all_data, 1, 0)  # (nb_channels, batch_size, h, w)
    # flatten height and width
    _data_per_channel = np.reshape(_data_per_channel, (_data_per_channel.shape[0], _data_per_channel.shape[1], -1))
    scatter_3d_generic(_data_per_channel[0], _data_per_channel[1], _data_per_channel[2],
                       all_labels, title=f"3D Latent Space of the First Three Dimensions", dir=opt.log_path,
                       file=f"_ 3D latent space idices 0_1_2", show=False, wandb_is_on=wandb_is_on,
                       label_idx_to_label_fn=translate_idx_to_label_fn)

    ### Histograms
    _, _, train_loader, _, test_loader, _ = get_dataloader.get_dataloader(dataset,
                                                                          purpose_is_unsupervised_learning=False)
    all_data, all_labels = _get_data_from_loader(train_loader, context_model, opt, num_datapoints=None)
    data_per_channel = np.moveaxis(all_data, 2, 0)  # (nb_channels, batch_size, seq_len)
    data_per_channel = np.reshape(data_per_channel, (data_per_channel.shape[0], data_per_channel.shape[1], -1))

    # plot histograms
    plot_histograms(opt, data_per_channel, f"SIM", max_dim=32, wandb_is_on=wandb_is_on)

    print("Finished")
    if USE_WANDB and wandb_is_on:
        wandb.finish()


if __name__ == '__main__':
    main()
