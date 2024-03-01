import torch
import numpy as np

from config_code.config_classes import OptionsConfig, DataSetConfig
from utils import utils


def val_by_latent_syllables(opt: DataSetConfig, device, dataloader, model, epoch, step):
    """
    Validate the training process by plotting the t-SNE
    representation of the latent space for different speakers
    """

    model.eval()

    # will contain batch of hidden representations for each module
    batch_latent_representations_per_module = []
    syllable_labels = []

    batch_size = opt.batch_size

    # one batch
    with torch.no_grad():
        _, (audios, _, pronounced_syllables, _) = next(enumerate(dataloader))
        model_input = audios.to(device)

        # iterate over layers
        for idx, layer in enumerate(model.module.fullmodel):
            # B x L x C
            (context, z) = layer.get_latents(model_input)

            model_input = z.permute(0, 2, 1)
            latent_rep = context.permute(0, 2, 1).cpu().numpy()

            batch_latent_representations_per_module.append(
                np.reshape(latent_rep, (batch_size, -1)))  # flatten

            pronounced_syllables = pronounced_syllables.numpy()
            syllable_labels.append(pronounced_syllables)

    # iterate over modules and plot t-SNE
    for idx, _ in enumerate(model.module.fullmodel):
        utils.fit_TSNE_and_plot(
            opt,
            batch_latent_representations_per_module[idx],
            syllable_labels[idx],
            f"{epoch}_{step}_model_{idx}",
        )

    model.train()
