from torch import Tensor
import torch
import torch.nn as nn

from config_code.config_classes import OptionsConfig
# https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py,
# https://medium.com/dataseries/convolutional-autoencoder-in-pytorch-on-mnist-dataset-d65145c132ac

from models import (
    cnn_encoder, autoregressor,
    loss_InfoNCE
)


class CPCIndependentModule(nn.Module):
    def __init__(
            self, opt: OptionsConfig,
            enc_kernel_sizes, enc_strides, enc_padding, nb_channels_cnn, nb_channels_regress,
            max_pool_k_size=None, max_pool_stride=None, calc_accuracy=False, prediction_step=12):
        super(CPCIndependentModule, self).__init__()

        self.opt = opt
        self.calc_accuracy = calc_accuracy
        self.nb_channels_cnn = nb_channels_cnn
        self.nb_channels_regressor = nb_channels_regress

        # encoder, out: B x L x C = (22, 55, 512)
        self.encoder: cnn_encoder.CNNEncoder = cnn_encoder.CNNEncoder(
            opt=opt,
            inp_nb_channels=1,
            out_nb_channels=nb_channels_cnn,
            kernel_sizes=enc_kernel_sizes,
            strides=enc_strides,
            padding=enc_padding,
            max_pool_k_size=max_pool_k_size,
            max_pool_stride=max_pool_stride,
        )

        self.autoregressor = autoregressor.Autoregressor(
            opt=opt, input_size=self.nb_channels_cnn, hidden_dim=self.nb_channels_regressor
        )

        # hidden dim of the encoder is the input dim of the loss
        self.loss = loss_InfoNCE.InfoNCE_Loss(
            opt, hidden_dim=self.nb_channels_regressor, enc_hidden=self.nb_channels_cnn, calc_accuracy=calc_accuracy,
            prediction_step=prediction_step)

    def get_latents(self, x) -> (Tensor, Tensor):
        z, _ = self.encoder(x)  # second param is for distributions (sigma), not used in CPC
        z = z.permute(0, 2, 1)  # (b, 55, 512)
        c = self.autoregressor(z)
        return c, z

    def forward(self, x):
        """
        combines all the operations necessary for calculating the loss and accuracy of the network given the input
        :param x: batch with sampled audios (dimensions: B x C x L)
        :return: total_loss - average loss over all samples, timesteps and prediction steps in the batch
                accuracies - average accuracies over all samples, timesteps and predictions steps in the batch
                c - latent representation of the input (either the output of the autoregressor,
                if use_autoregressor=True, or the output of the encoder otherwise)
        """

        # B x L x C = Batch size x #channels x length
        c, z = self.get_latents(x)

        nce_loss, accuracies = self.loss.get_loss(z, c)
        kld_loss = torch.tensor(0.0, device=self.opt.device)
        total_loss = nce_loss

        # for multi-GPU training
        total_loss = total_loss.unsqueeze(0)
        accuracies = accuracies.unsqueeze(0)

        nce_loss = nce_loss.unsqueeze(0)
        kld_loss = kld_loss.unsqueeze(0)

        return total_loss, accuracies, z, nce_loss, kld_loss
