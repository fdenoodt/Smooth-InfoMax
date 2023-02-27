import torch.nn as nn


class CNNEncoder(nn.Module):
    def __init__(self, inp_nb_channels, out_nb_channels, kernel_sizes, strides, padding):
        super(CNNEncoder, self).__init__()

        self.nb_channels = out_nb_channels

        assert (
            len(kernel_sizes) == len(strides) == len(padding)
        ), "Inconsistent size of network parameters (kernels, strides and padding)"

        self.model = nn.Sequential()

        for idx, _ in enumerate(kernel_sizes):
            self.model.add_module(
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

    def forward(self, x):
        return self.model(x)