import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden, kernel_sizes, strides, padding):
        super(CNNEncoder, self).__init__()

        self.hidden = hidden

        assert (
            len(kernel_sizes) == len(strides) == len(padding)
        ), "Inconsistent size of network parameters (kernels, strides and padding)"

        self.model = nn.Sequential()

        for idx, _ in enumerate(kernel_sizes):
            self.model.add_module(
                f"layer {idx}",
                self.new_block(
                    input_dim,
                    self.hidden,
                    kernel_sizes[idx],
                    strides[idx],
                    padding[idx],
                ),
            )
            input_dim = self.hidden

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

    def forward_through_n_layers(self, x, n):
        for i in range(n):
            x = self.model[i](x)
        return x
