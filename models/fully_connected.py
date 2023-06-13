import torch.nn as nn


class FullyConnected(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FullyConnected, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )

    def forward(self, x):

        # flatten the input
        print(x.size())
        print(x.shape)
        x = x.view(x.size(0), -1)
        return self.model(x)
