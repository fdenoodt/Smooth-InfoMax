import torch
import torch.nn as nn

from models import independent_module
from utils import utils


class FullModel(nn.Module):
    def __init__(
        self,
        opt,
        kernel_sizes,
        strides,
        padding,
        enc_hidden,
        calc_accuracy=False,
    ):
        """
        Entire CPC model that can be split into smaller chunks for training
        """
        super(FullModel, self).__init__()

        self.opt = opt
        self.enc_hidden = enc_hidden

        # load model
        self.fullmodel = nn.ModuleList([])

        if self.opt["model_splits"] == 1:
            # CPC model
            self.fullmodel.append(
                independent_module.IndependentModule(
                    opt,
                    enc_kernel_sizes=kernel_sizes,
                    enc_strides=strides,
                    enc_padding=padding,
                    hidden_dim=enc_hidden,
                    calc_accuracy=calc_accuracy,
                )
            )
        elif (
            self.opt["model_splits"] == 5
        ):  # GIM model in which the last autoregressive layer is trained independently
            enc_input = 1
            for i, _ in enumerate(kernel_sizes):
                self.fullmodel.append(
                    # enc_padding, hidden_dim, calc_accuracy=False,
                    independent_module.IndependentModule(
                        opt,
                        enc_input=enc_input,
                        enc_kernel_sizes=[kernel_sizes[i]],
                        enc_strides=[strides[i]],
                        enc_padding=[padding[i]],
                        hidden_dim=enc_hidden,
                        calc_accuracy=calc_accuracy,
                    )
                )
                enc_input = enc_hidden
        else:
            raise Exception("Invalid option for opt['model_splits']")

    def forward(self, x):
        model_input = x

        cur_device = utils.get_device(self.opt, x)

        # first dimension is used for concatenating results from different GPUs
        loss = torch.zeros(1, len(self.fullmodel), device=cur_device)
        accuracy = torch.zeros(1, len(self.fullmodel), device=cur_device)

        for idx, layer in enumerate(self.fullmodel):
            loss[:, idx], accuracy[:, idx], z = layer(model_input)
            model_input = z.permute(0, 2, 1).detach()

        return loss
    