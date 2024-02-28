import torch
import torch.nn as nn

from config_code.config_classes import OptionsConfig
from config_code.architecture_config import ArchitectureConfig, ModuleConfig
from models import independent_module, independent_module_regressor, independent_module_cpc
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

        self.fullmodel: nn.ModuleList = nn.ModuleList([])
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
        cpc_module = independent_module_cpc.CPCIndependentModule(
            opt,
            enc_kernel_sizes=m.kernel_sizes,
            enc_strides=m.strides,
            enc_padding=m.padding,
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
        padding = module_config.padding
        cnn_hidden_dim = module_config.cnn_hidden_dim

        assert module_config.regressor_hidden_dim is not None  # TODO: implement regressor_hidden_dim
        regressor_hidden_dim = module_config.regressor_hidden_dim

        max_pool_k_size = module_config.max_pool_k_size
        max_pool_stride = module_config.max_pool_stride
        prediction_step = module_config.prediction_step

        module = independent_module.IndependentModule(
            opt,
            enc_input=1 if is_first_module else cnn_hidden_dim,  # 1 if first layer, else cnn_hidden_dim
            enc_kernel_sizes=kernel_sizes,  # [10, 8, 4, 4, 4]
            enc_strides=strides,  # [5, 4, 2, 2, 2]
            enc_padding=padding,  # [2, 2, 2, 2, 1]
            nb_channels_cnn=cnn_hidden_dim,  # 512
            nb_channels_regress=regressor_hidden_dim,  # 256
            max_pool_k_size=max_pool_k_size,
            max_pool_stride=max_pool_stride,
            calc_accuracy=calc_accuracy,
            prediction_step=prediction_step,
            predict_distributions=module_config.predict_distributions
        )
        return module

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

    def forward_through_all_cnn_modules(self, x):
        model_input = x

        for idx, layer in enumerate(self.fullmodel):
            if idx + 1 < len(self.fullmodel):
                _, z = layer.get_latents(model_input)
                model_input = z.permute(0, 2, 1)

        return model_input
