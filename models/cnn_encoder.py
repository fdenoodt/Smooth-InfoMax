from typing import List, Tuple
from torch import Tensor
import torch.nn as nn
import torch


class CNNEncoder(nn.Module):
    def __init__(self, opt, inp_nb_channels, out_nb_channels, kernel_sizes, strides, padding, max_pool_k_size=None,
                 max_pool_stride=None):
        super(CNNEncoder, self).__init__()

        self.opt = opt
        self.nb_channels = out_nb_channels

        assert (
                len(kernel_sizes) == len(strides) == len(padding)
        ), "Inconsistent size of network parameters (kernels, strides and padding)"

        self.encoder: nn.Sequential = nn.Sequential()
        self.encoder_mu: nn.Conv1d = nn.Conv1d(out_nb_channels, out_nb_channels, kernel_size=1, stride=1, padding=0)
        self.encoder_var: nn.Conv1d = nn.Conv1d(out_nb_channels, out_nb_channels, kernel_size=1, stride=1, padding=0)

        # add the layers to self.encoder
        for idx, _ in enumerate(kernel_sizes):
            self.encoder.add_module(
                f"layer {idx}",
                CNNEncoder.new_block(
                    inp_nb_channels,
                    self.nb_channels,
                    kernel_sizes[idx],
                    strides[idx],
                    padding[idx],
                ),
            )
            inp_nb_channels = self.nb_channels

            if max_pool_k_size:
                assert max_pool_stride, "max_pool_stride must be set if max_pool_k_size is set"

                # add maxpool to encoder
                self.encoder.add_module(
                    f"maxpool {idx}", nn.MaxPool1d(max_pool_k_size, max_pool_stride))

    @staticmethod
    def new_block(in_dim, out_dim, kernel_size, stride, padding):
        new_block = nn.Sequential(
            CNNEncoder.conv1d(in_dim, out_dim, kernel_size, stride, padding),
            nn.ReLU()
        )
        return new_block

    @staticmethod
    def conv1d(in_dim, out_dim, kernel_size, stride, padding):
        return nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x) -> Tuple[Tensor, Tensor]:
        # x is batch of audio files of shape [N x C x L]
        result = self.encoder(x)
        mu = self.encoder_mu(result)
        log_var = self.encoder_var(result)

        assert mu.shape == log_var.shape == result.shape, f"mu shape: {mu.shape}, log_var shape: {log_var.shape}, result shape: {result.shape}"
        return mu, log_var
