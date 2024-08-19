from typing import List, Tuple
import torch
import torch.nn as nn
from torch import Tensor

from config_code.config_classes import OptionsConfig


class CNNEncoder(nn.Module):
    def __init__(self, opt: OptionsConfig, inp_nb_channels, out_nb_channels, kernel_sizes, strides, padding,
                 relus: List[bool],
                 max_pool_k_size=None, max_pool_stride=None):
        super(CNNEncoder, self).__init__()

        self.opt = opt
        self.nb_channels = out_nb_channels

        assert (
                len(kernel_sizes) == len(strides) == len(padding)
        ), "Inconsistent size of network parameters (kernels, strides and padding)"

        self.encoder = []  # Change to list
        self.encoder_mu: nn.Conv1d = nn.Conv1d(out_nb_channels, out_nb_channels, kernel_size=1, stride=1, padding=0)
        self.encoder_var: nn.Conv1d = nn.Conv1d(out_nb_channels, out_nb_channels, kernel_size=1, stride=1, padding=0)

        # add the layers to self.encoder
        for idx, _ in enumerate(kernel_sizes):
            self.encoder.append(
                CNNEncoder.new_block(
                    inp_nb_channels,
                    self.nb_channels,
                    kernel_sizes[idx],
                    strides[idx],
                    padding[idx],
                    relus[idx],
                    bn=opt.encoder_config.use_batch_norm
                )
            )
            inp_nb_channels = self.nb_channels

            if max_pool_k_size:
                assert max_pool_stride, "max_pool_stride must be set if max_pool_k_size is set"

                # add maxpool to encoder
                self.encoder.append(nn.MaxPool1d(max_pool_k_size, max_pool_stride))

        self.encoder = nn.ModuleList(self.encoder)  # Convert list to ModuleList

    @staticmethod
    def new_block(in_dim, out_dim, kernel_size, stride, padding, relu: bool, bn: bool):
        new_block = CNNEncoder.conv1d(in_dim, out_dim, kernel_size, stride, padding)
        if relu:  # always True, except for special case in CPC for density experiments such that equal number of layers for easier comparison with SIM/GIM
            # new_block = nn.Sequential(new_block, nn.ReLU())
            # also batchnorm
            if bn:
                new_block = nn.Sequential(new_block, nn.BatchNorm1d(out_dim), nn.ReLU())
            else:
                new_block = nn.Sequential(new_block, nn.ReLU())
        return new_block

    @staticmethod
    def conv1d(in_dim, out_dim, kernel_size, stride, padding):
        return nn.Conv1d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x) -> Tuple[Tensor, Tensor]:
        # x is batch of audio files of shape [N x C x L]
        for layer in self.encoder:
            x = layer(x)
        mu = self.encoder_mu(x)
        log_var = self.encoder_var(x)

        assert mu.shape == log_var.shape == x.shape, f"mu shape: {mu.shape}, log_var shape: {log_var.shape}, result shape: {x.shape}"
        return mu, log_var

    def forward_intermediate_layer(self, x, layer_idx) -> Tuple[Tensor, Tensor]:
        """
        Forward pass until layer_idx and return the result. Returns 2 args due to plausible reparameterization trick.
        """
        if layer_idx == len(self.encoder) or layer_idx == -1:
            return self.forward(x)

        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            if idx == layer_idx:
                log_var_placeholder = torch.zeros_like(x)
                return x, log_var_placeholder

        raise ValueError(f"layer_idx {layer_idx} is out of range")
