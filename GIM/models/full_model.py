import torch
import torch.nn as nn

from models import independent_module, independent_module_regressor
from utils import utils


class FullModel(nn.Module):
    def __init__(
        self,
        opt,
        kernel_sizes,
        strides,
        padding,
        cnn_hidden_dim,
        regressor_hidden_dim,
        calc_accuracy=False,
    ):
        """
        Entire CPC model that can be split into smaller chunks for training
        """
        super(FullModel, self).__init__()

        self.opt = opt

        # load model
        self.fullmodel = nn.ModuleList([])
    
        if self.opt["model_splits"] == 1:
            # CPC model
            self.fullmodel.append(
                independent_module.IndependentModule(
                    opt,
                    enc_kernel_sizes=kernel_sizes, # [10, 8, 4, 4, 4]
                    enc_strides=strides, # [5, 4, 2, 2, 2]
                    enc_padding=padding, # [2, 2, 2, 2, 1]
                    nb_channels_cnn=cnn_hidden_dim, # 512
                    nb_channels_regress=regressor_hidden_dim, # 256
                    calc_accuracy=calc_accuracy,
                )
            )
        elif (
            self.opt["model_splits"] == 6
        ):  # GIM model in which the last autoregressive layer is trained independently
            assert opt['auto_regressor_after_module'] is False, "This option is not supported for model_splits == 6"
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
                        nb_channels_cnn=cnn_hidden_dim,
                        nb_channels_regress=regressor_hidden_dim,
                        calc_accuracy=calc_accuracy,
                    )
                )
                enc_input = cnn_hidden_dim

            # Just regressor layer
            self.fullmodel.append(
                independent_module_regressor.AutoregressorIndependentModule(
                    opt,
                    nb_channels_cnn=cnn_hidden_dim,
                    nb_channels_regress=regressor_hidden_dim,
                    calc_accuracy=calc_accuracy,
                )
            )
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
            model_input = z.permute(0, 2, 1).detach()  # (22, 55, 512)

        return loss
