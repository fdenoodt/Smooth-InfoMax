from typing import List

import torch
import torch.nn as nn

from config_code.config_classes import OptionsConfig, Dataset
from config_code.architecture_config import ArchitectureConfig, ModuleConfig
from models import independent_module, independent_module_regressor, independent_module_cpc
from models.abstract_module import AbstractModule
from utils import utils


class FullModel(nn.Module):
    def __init__(
            self,
            opt: OptionsConfig,
            calc_accuracy=False,
    ):
        """
        Entire CPC model that can be split into smaller chunks for training
        """
        super(FullModel, self).__init__()

        self.fullmodel: nn.ModuleList[AbstractModule] = nn.ModuleList([])
        self.opt: OptionsConfig = opt
        self.output_dim: int = opt.encoder_config.architecture.modules[-1].regressor_hidden_dim

        architecture: ArchitectureConfig = opt.encoder_config.architecture
        # CNN modules
        for idx, module_config in enumerate(architecture.modules):
            # only relevant for replicating the CPC model, not for Greedy InfoMax or Smooth Infomax
            if module_config.is_cnn_and_autoregressor:
                assert len(architecture.modules) == 1
                m = module_config
                self.fullmodel.append(self.cpc_module_from_config(opt, m, calc_accuracy))


            # Auto-regressor module
            elif module_config.is_autoregressor:
                m = module_config
                # assert no distributions
                assert not m.predict_distributions, "Distributions not implemented for autoregressor"
                self.fullmodel.append(
                    independent_module_regressor.AutoregressorIndependentModule(
                        opt,
                        nb_channels_cnn=m.cnn_hidden_dim,
                        nb_channels_regress=m.regressor_hidden_dim,
                        calc_accuracy=calc_accuracy,
                        prediction_step=m.prediction_step))

            # Regular module (CNN)
            else:
                indep_module = FullModel.cnn_module_from_config(opt, module_config, calc_accuracy, idx == 0)
                self.fullmodel.append(indep_module)

    @staticmethod
    def cpc_module_from_config(opt, m: ModuleConfig, calc_accuracy) -> independent_module_cpc.CPCIndependentModule:
        nb_channels_inp = FullModel.get_nb_channels_inp(opt, is_first_module=True, cnn_hidden_dim=m.cnn_hidden_dim)
        cpc_module = independent_module_cpc.CPCIndependentModule(
            opt,
            nb_channels_inp=nb_channels_inp,
            enc_kernel_sizes=m.kernel_sizes,
            enc_strides=m.strides,
            enc_paddings=m.padding,
            enc_non_linearities=m.non_linearities,
            nb_channels_cnn=m.cnn_hidden_dim,
            nb_channels_regress=m.regressor_hidden_dim,
            max_pool_k_size=m.max_pool_k_size,
            max_pool_stride=m.max_pool_stride,
            calc_accuracy=calc_accuracy,
            prediction_step=m.prediction_step,
        )
        return cpc_module

    @staticmethod
    def cnn_module_from_config(opt, module_config, calc_accuracy, is_first_module) \
            -> independent_module.IndependentModule:
        kernel_sizes = module_config.kernel_sizes
        strides = module_config.strides
        paddings = module_config.padding
        non_linearities = module_config.non_linearities
        cnn_hidden_dim = module_config.cnn_hidden_dim

        assert module_config.regressor_hidden_dim is not None  # TODO: implement regressor_hidden_dim
        regressor_hidden_dim = module_config.regressor_hidden_dim

        max_pool_k_size = module_config.max_pool_k_size
        max_pool_stride = module_config.max_pool_stride
        prediction_step = module_config.prediction_step

        nb_channels_inp = FullModel.get_nb_channels_inp(opt, is_first_module, cnn_hidden_dim)
        module = independent_module.IndependentModule(
            opt,
            nb_channels_inp=nb_channels_inp,  # 1 if first layer, else cnn_hidden_dim
            enc_kernel_sizes=kernel_sizes,  # [10, 8, 4, 4, 4]
            enc_strides=strides,  # [5, 4, 2, 2, 2]
            enc_paddings=paddings,  # [2, 2, 2, 2, 1]
            enc_non_linearities=non_linearities,  # [True, True, True, True, True]
            nb_channels_cnn=cnn_hidden_dim,  # 512
            nb_channels_regress=regressor_hidden_dim,  # 256
            max_pool_k_size=max_pool_k_size,
            max_pool_stride=max_pool_stride,
            calc_accuracy=calc_accuracy,
            prediction_step=prediction_step,
            predict_distributions=module_config.predict_distributions
        )
        return module

    @staticmethod
    def get_nb_channels_inp(opt: OptionsConfig, is_first_module: bool, cnn_hidden_dim: int):
        if is_first_module:
            # Radio: 2
            # Audio: 1 channel
            nb_channels_inp = 2 if opt.encoder_config.dataset.dataset == Dataset.RADIO else 1
        else:
            nb_channels_inp = cnn_hidden_dim
        return nb_channels_inp

    def forward(self, x):
        model_input = x

        cur_device = utils.get_device(self.opt, x)

        # first dimension is used for concatenating results from different GPUs
        loss = torch.zeros(1, len(self.fullmodel), device=cur_device)
        nce_loss = torch.zeros(1, len(self.fullmodel), device=cur_device)
        kld_loss = torch.zeros(1, len(self.fullmodel), device=cur_device)
        accuracy = torch.zeros(1, len(self.fullmodel), device=cur_device)

        for idx, layer in enumerate(self.fullmodel):
            loss[:, idx], accuracy[:, idx], z, nce_loss[:, idx], kld_loss[:, idx] = layer(model_input)
            model_input = z.permute(0, 2, 1).detach()

        return loss, nce_loss, kld_loss

    def forward_through_all_modules(self, x):
        model_input = x

        for idx, layer in enumerate(self.fullmodel):
            if idx + 1 < len(self.fullmodel):
                _, z = layer.get_latents(model_input)
                model_input = z.permute(0, 2, 1)

        context, _ = self.fullmodel[idx].get_latents(model_input)
        return context

    def _forward_through_module(self, x, stop_idx):
        if stop_idx == -1:  # take last cnn module
            stop_idx = (len(self.fullmodel) - 1) - 1  # skip the regressor

        model_input = x
        assert stop_idx <= len(self.fullmodel) - 1, \
            f"stop_idx={stop_idx} is larger than the number of modules in the model"

        for idx, layer in enumerate(self.fullmodel[:-1]):  # skip the regressor
            if idx <= stop_idx:
                _, z = layer.get_latents(model_input)
                model_input = z.permute(0, 2, 1)

        return model_input

    def forward_through_all_cnn_modules(self, x):
        return self._forward_through_module(x, len(self.fullmodel) - 1)  # skip the regressor

    def forward_through_module(self, x, idx):
        """Foward through all modules until the target module (inclusive)"""
        return self._forward_through_module(x, idx)

    def forward_through_layer(self, x, module_idx, layer_idx):
        """
        Forward through a specific layer in a specific module
        """
        nb_modules = len(self.fullmodel)
        model_input = x
        for idx, module in enumerate(self.fullmodel[:nb_modules - 1]):  # until target module (exclusive)
            module: AbstractModule = module  # type hinting

            if idx <= module_idx - 1:  # stop one module before the target module
                _, z = module.get_latents(model_input)
                model_input = z.permute(0, 2, 1)

        module = self.fullmodel[module_idx]  # target module
        _, z = module.get_latents_of_intermediate_layers(model_input, layer_idx)
        model_input = z.permute(0, 2, 1)

        return model_input
