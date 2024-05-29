import torch
import torch.nn as nn

from config_code.config_classes import ClassifierConfig


class ClassificationModel(torch.nn.Module):
    def __init__(self, classifier_config: ClassifierConfig, in_channels=256, num_classes=200, hidden_nodes=0):
        super().__init__()
        self.in_channels = in_channels
        self.avg_pool = nn.AvgPool2d((7, 7), padding=0)
        self.model = nn.Sequential()

        if hidden_nodes > 0:
            raise Exception("Hidden nodes not implemented yet")

            self.model.add_module(
                "layer1", nn.Linear(self.in_channels, hidden_nodes, bias=True)
            )

            self.model.add_module("ReLU", nn.ReLU())
            self.model.add_module("Dropout", nn.Dropout(p=0.5))

            self.model.add_module(
                "layer 2", nn.Linear(hidden_nodes, num_classes, bias=True)
            )

        else:
            bias = classifier_config.bias
            self.model.add_module(
                "layer1", nn.Linear(self.in_channels, num_classes, bias=bias)
            )

        print(self.model)

    def forward(self, x, *args):
        x = self.avg_pool(x).squeeze()
        x = self.model(x).squeeze()
        return x
