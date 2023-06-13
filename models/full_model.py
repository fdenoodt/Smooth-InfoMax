import torch
import torch.nn as nn

from models import independent_module, independent_module_regressor
from utils import utils


class FullModel(nn.Module):
    def __init__(
        self,
        opt,
        calc_accuracy=False,
    ):
        """
        Entire CPC model that can be split into smaller chunks for training
        """
        super(FullModel, self).__init__()

        self.opt = opt
        assert self.opt["model_splits"] in [
            1, 2, 3], "Invalid option for opt['model_splits']"

        architecture = opt["architecture_module_1"]
        kernel_sizes = architecture["kernel_sizes"]
        strides = architecture["strides"]
        padding = architecture["padding"]
        cnn_hidden_dim = architecture["cnn_hidden_dim"]
        regressor_hidden_dim = architecture["regressor_hidden_dim"]
        max_pool_k_size = architecture["max_pool_k_size"]
        max_pool_stride = architecture["max_pool_stride"]
        prediction_step = architecture["prediction_step"]

        # load model
        self.fullmodel = nn.ModuleList([])

        self.fullmodel.append(
            independent_module.IndependentModule(
                opt,
                enc_kernel_sizes=kernel_sizes,  # [10, 8, 4, 4, 4]
                enc_strides=strides,  # [5, 4, 2, 2, 2]
                enc_padding=padding,  # [2, 2, 2, 2, 1]
                nb_channels_cnn=cnn_hidden_dim,  # 512
                nb_channels_regress=regressor_hidden_dim,  # 256
                max_pool_k_size=max_pool_k_size,
                max_pool_stride=max_pool_stride,
                calc_accuracy=calc_accuracy,
                prediction_step=prediction_step,
            )
        )

        if self.opt["model_splits"] >= 2:
            enc_input = cnn_hidden_dim

            architecture = opt["architecture_module_2"]
            kernel_sizes = architecture["kernel_sizes"]
            strides = architecture["strides"]
            padding = architecture["padding"]
            cnn_hidden_dim = architecture["cnn_hidden_dim"]
            regressor_hidden_dim = architecture["regressor_hidden_dim"]
            max_pool_k_size = architecture["max_pool_k_size"]
            max_pool_stride = architecture["max_pool_stride"]
            prediction_step = architecture["prediction_step"]

            self.fullmodel.append(
                independent_module.IndependentModule(
                    opt,
                    enc_input=enc_input,
                    enc_kernel_sizes=kernel_sizes,
                    enc_strides=strides,
                    enc_padding=padding,
                    nb_channels_cnn=cnn_hidden_dim,
                    nb_channels_regress=regressor_hidden_dim,
                    max_pool_k_size=max_pool_k_size,
                    max_pool_stride=max_pool_stride,
                    calc_accuracy=calc_accuracy,
                    prediction_step=prediction_step,
                )
            )

        if self.opt["model_splits"] == 3:  # append auto-regressor
            self.fullmodel.append(
                independent_module_regressor.AutoregressorIndependentModule(
                    opt,
                    nb_channels_cnn=cnn_hidden_dim,
                    nb_channels_regress=regressor_hidden_dim,
                    calc_accuracy=calc_accuracy,
                    prediction_step=prediction_step))

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
