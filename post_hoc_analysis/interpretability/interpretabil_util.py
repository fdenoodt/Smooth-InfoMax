import random
from typing import Union

import numpy as np

from config_code.config_classes import OptionsConfig

from PIL import Image
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import wandb
from utils.helper_functions import colour_palette_vowels, translate_vowel_number_to_vowel, histogram, scatter_syllable


def plot_tsne_syllable(opt: OptionsConfig, feature_space: np.ndarray, label_indices: np.ndarray, gim_name: str,
                       lr: Union[float, str], n_iter: int, perplexity: int, wandb_is_on: bool):
    """
    SAVES t-SNE plot to the log directory AND logs it to wandb if wandb_is_on is True
    """
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

    scatter_syllable(projection, label_indices,
                     title=f"t-SNE Latent space - {gim_name}", dir=save_dir, file=file, show=False)

    print(f"Saved t-SNE plot to {save_dir}/{file}")

    if wandb_is_on:
        wandb.log({f"LatSpace/t-SNE_latent_space_{gim_name}": [wandb.Image(f"{save_dir}/{file}")]})


def plot_histograms(opt: OptionsConfig, feature_space_per_channel, gim_name, max_dim: int, wandb_is_on: bool):
    # feature_space_per_channel: (nb_channels, batch_size, seq_len)
    save_dir = opt.log_path
    images = []

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
            images.append(Image.open(f"{save_dir}/{file}.png"))

    # Create a collage of images and log it to wandb
    if wandb_is_on:
        collage = Image.new('RGB', (images[0].width * len(images), images[0].height))
        for i, image in enumerate(images):
            collage.paste(image, (i * images[0].width, 0))
        wandb.log({f"LatSpace/distribution_latent_space_{gim_name}": [wandb.Image(collage)]})


def scatter_3d(x, y, z, labels, title, dir, file, show, wandb_is_on):
    # x, y, z: (batch_size, seq_len)
    # labels: (batch_size,) so must copy the labels for each seq_len
    if len(x.shape) > 1:
        labels = np.repeat(labels, x.shape[1])

    x, y, z = x.flatten(), y.flatten(), z.flatten()

    # limit to 1000 points
    if x.shape[0] > 5000:
        indices = random.sample(range(x.shape[0]), 1000)
        x, y, z, labels = x[indices], y[indices], z[indices], labels[indices]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    palette = colour_palette_vowels()

    for i, vowel_idx in enumerate(np.unique(labels)):
        indices = np.where(labels == vowel_idx)
        color = np.tile(palette[i], (len(indices), 1))
        ax.scatter(x[indices], y[indices], z[indices], c=color, label=translate_vowel_number_to_vowel(vowel_idx))

    plt.legend()

    # set x, y, z limits between -3 and 3
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_zlim(-2.2, 2.2)

    ax.set_title(title)

    if show:
        plt.show()

    fig.savefig(f"{dir}/{file}.png")
    fig.savefig(f"{dir}/{file}.pdf")

    if wandb_is_on:
        wandb.log({f"LatSpace/3D_latent_space_{file}": [wandb.Image(f"{dir}/{file}.png")]})

    return f"{dir}/{file}.png"
