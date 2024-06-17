from abc import ABC, abstractmethod
from torch import nn, Tensor


class AbstractModule(nn.Module, ABC):
    def __init__(self):
        super(AbstractModule, self).__init__()

    @abstractmethod
    def get_latents(self, x) -> (Tensor, Tensor):
        pass

    @abstractmethod
    def forward(self, x) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
        pass

    @abstractmethod
    def get_latents_of_intermediate_layers(self, x, layer_idx) -> (Tensor, Tensor):
        pass
