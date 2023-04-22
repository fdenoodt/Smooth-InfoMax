from typing import List
from torch import Tensor
import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, opt, inp_nb_channels, out_nb_channels, kernel_sizes, strides, padding, max_pool_k_size=None, max_pool_stride=None):
        super(CNNEncoder, self).__init__()

        self.opt = opt
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
                # self.encoder_mu = self.new_block(
                #     inp_nb_channels, self.nb_channels, kernel_sizes[idx], strides[idx], padding[idx],)

                self.encoder_mu = nn.Sequential(nn.Conv1d(
                    inp_nb_channels, self.nb_channels, kernel_sizes[idx], strides[idx], padding[idx]),
                    nn.ReLU()) # TODO: TEMPORARY

                self.encoder_var = nn.Sequential(nn.Conv1d(
                    inp_nb_channels, self.nb_channels, kernel_sizes[idx], strides[idx], padding[idx]),
                    nn.ReLU()) # TODO: TEMPORARY
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

                if max_pool_k_size:
                    assert max_pool_stride, "max_pool_stride must be set if max_pool_k_size is set"

                    # add maxpool to encoder
                    self.encoder.add_module(
                        f"maxpool {idx}", nn.MaxPool1d(max_pool_k_size, max_pool_stride))

            inp_nb_channels = self.nb_channels

    def new_block(self, in_dim, out_dim, kernel_size, stride, padding):
        new_block = nn.Sequential(
            nn.Conv1d(
                in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.ReLU()
            # nn.Sigmoid()
        )
        return new_block

    def forward(self, x) -> List[Tensor]:
        # x is batch of audio files of shape [N x C x L]
        result = self.encoder(x)

        mu = self.encoder_mu(result)
        log_var = self.encoder_var(result)
        return [mu, log_var]
