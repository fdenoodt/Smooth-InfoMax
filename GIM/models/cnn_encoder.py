from typing import List
from torch import Tensor
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, inp_nb_channels, out_nb_channels, kernel_sizes, strides, padding):
        super(CNNEncoder, self).__init__()

        self.nb_channels = out_nb_channels

        assert (
            len(kernel_sizes) == len(strides) == len(padding)
        ), "Inconsistent size of network parameters (kernels, strides and padding)"

        self.encoder = nn.Sequential()
        self.encoder_mu = None
        self.encoder_var = None

        # add the layers
        for idx, _ in enumerate(kernel_sizes):
            # if at last layer add the mu and var layers
            if idx == len(kernel_sizes) - 1:
                self.encoder_mu = self.new_block(
                    inp_nb_channels, self.nb_channels, kernel_sizes[idx], strides[idx], padding[idx],)

                self.encoder_var = self.new_block(
                    inp_nb_channels, self.nb_channels, kernel_sizes[idx], strides[idx], padding[idx])
            else:
                self.encoder.add_module(
                    f"layer {idx}",
                    self.new_block(
                        inp_nb_channels,
                        self.nb_channels,
                        kernel_sizes[idx],
                        strides[idx],
                        padding[idx],
                    ),
                )

            inp_nb_channels = self.nb_channels

    def new_block(self, in_dim, out_dim, kernel_size, stride, padding):
        new_block = nn.Sequential(
            nn.Conv1d(
                in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.ReLU(),
        )
        return new_block

    # from vae:
    # self.encoder_cnn = nn.Sequential(
    #     nn.Conv2d(1, 8, 3, stride=2, padding=1),
    #     nn.ReLU(True),
    #     nn.Conv2d(8, 16, 3, stride=2, padding=1),
    #     nn.BatchNorm2d(16),
    #     nn.ReLU(True),
    #     nn.Conv2d(16, 32, 3, stride=2, padding=0),
    #     nn.ReLU(True)
    # )

    # # In: [b, 32, 3, 3]
    # # Out: [b, latent_dim, 3, 3]
    #
    # self.cnn_mu = nn.Sequential(
    #     nn.Conv2d(32, latent_dim, 3, stride=1, padding=1),
    #     nn.ReLU(True))

    # self.cnn_var = nn.Sequential(
    #     nn.Conv2d(32, latent_dim, 3, stride=1, padding=1),
    #     nn.ReLU(True))
    # def encode(self, input: Tensor) -> List[Tensor]:
    #     """
    #     Encodes the input by passing through the encoder network
    #     and returns the latent codes.
    #     :param input: (Tensor) Input tensor to encoder [N x C x H x W]
    #     :return: (Tensor) List of latent codes
    #     """
    #     result = self.encoder_cnn(input)
    #     # out: [b x 32 x 3 x 3] = b x c x h x w

    #     # Split the result into mu and var components
    #     # of the latent Gaussian distribution
    #     mu = self.cnn_mu(result) # out: [b x latent_dim x 3 x 3]
    #     log_var = self.cnn_var(result)

    #     return [mu, log_var]

    def forward(self, x) -> List[Tensor]:
        result = self.encoder(x)
        mu = self.encoder_mu(result)
        log_var = self.encoder_var(result)
        return [mu, log_var]
