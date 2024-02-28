import torch.nn as nn

from config_code.architecture_config import DecoderArchitectureConfig


class Decoder(nn.Module):
    def __init__(self, decoder_architecture: DecoderArchitectureConfig):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential()
        nb_layers = len(decoder_architecture.kernel_sizes)
        for i in range(nb_layers):
            self.decoder.add_module(
                f"conv_transpose_{i}",
                nn.ConvTranspose1d(
                    in_channels=decoder_architecture.input_dim if i == 0 else decoder_architecture.hidden_dim,
                    out_channels=decoder_architecture.hidden_dim if i < nb_layers - 1 else decoder_architecture.output_dim,
                    kernel_size=decoder_architecture.kernel_sizes[i],
                    stride=decoder_architecture.strides[i],
                    padding=decoder_architecture.paddings[i],
                    output_padding=decoder_architecture.output_paddings[i],
                ),
            )
            if i < nb_layers - 1:
                self.decoder.add_module(
                    f"relu_{i}",
                    nn.ReLU(),
                )

    def forward(self, x):
        return self.decoder(x)


if __name__ == "__main__":
    pass
