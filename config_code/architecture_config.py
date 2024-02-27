from typing import Optional, List


class ModuleConfig:
    def __init__(self,
                 max_pool_k_size: Optional[int], max_pool_stride: Optional[int], kernel_sizes: list,
                 strides: list, padding: list, cnn_hidden_dim: int, regressor_hidden_dim: Optional[int],
                 is_autoregressor: bool, prediction_step: int, predict_distributions: bool,
                 is_cnn_and_autoregressor: Optional[bool] = False): # for CPC
        assert len(kernel_sizes) == len(strides) == len(padding)

        if is_autoregressor and not (is_cnn_and_autoregressor):
            # when a typical regressor module from Sindy (then no cnn layer)
            assert len(kernel_sizes) == 0
            assert len(strides) == 0
            assert len(padding) == 0
            assert max_pool_k_size is None
            assert max_pool_stride is None

        self.max_pool_k_size = max_pool_k_size
        self.max_pool_stride = max_pool_stride
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.padding = padding
        self.cnn_hidden_dim = cnn_hidden_dim
        self.regressor_hidden_dim = regressor_hidden_dim
        self.prediction_step = prediction_step  # eg 12
        self.predict_distributions = predict_distributions
        self.is_autoregressor = is_autoregressor
        self.is_cnn_and_autoregressor = is_cnn_and_autoregressor  # relevant in CPC, where a single module has both cnn and autoregressor layers

    def __str__(self):
        return f"ModuleConfig(max_pool_k_size={self.max_pool_k_size}, max_pool_stride={self.max_pool_stride}, " \
               f"kernel_sizes={self.kernel_sizes}, strides={self.strides}, padding={self.padding}, " \
               f"cnn_hidden_dim={self.cnn_hidden_dim}, regressor_hidden_dim={self.regressor_hidden_dim}, " \
               f"prediction_step={self.prediction_step}, predict_distributions={self.predict_distributions}, " \
               f"is_autoregressor={self.is_autoregressor}"

    @staticmethod
    def get_modules_from_list(kernel_sizes, strides, paddings, cnn_hidden_dim, predict_distribution):
        """
        Splits each layer into a separate module
        """
        assert len(kernel_sizes) == len(strides) == len(paddings)
        # create module per kernel size
        modules = [
            ModuleConfig(
                max_pool_k_size=None,
                max_pool_stride=None,
                kernel_sizes=[kernel_sizes[i]],
                strides=[strides[i]],
                padding=[paddings[i]],
                cnn_hidden_dim=cnn_hidden_dim,
                is_autoregressor=False,
                regressor_hidden_dim=cnn_hidden_dim,
                prediction_step=12,
                predict_distributions=predict_distribution
            )
            for i in range(len(kernel_sizes))
        ]
        return modules


class ArchitectureConfig:  # for encoder
    def __init__(self, modules: list[ModuleConfig], is_cpc: Optional[bool] = False):
        if is_cpc:
            assert len(modules) == 1

        self.is_cpc = is_cpc
        self.modules: list[ModuleConfig] = modules

    def __str__(self):
        modules: str = ", ".join([str(module) for module in self.modules])
        return f"ArchitectureConfig(modules={modules})"


class DecoderArchitectureConfig:
    def __init__(self, kernel_sizes: List[int], strides: List[int], paddings: List[int], output_paddings: List[int],
                 input_dim: int, hidden_dim: int, output_dim: int):
        assert len(kernel_sizes) == len(strides) == len(paddings)
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.output_paddings = output_paddings
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def __str__(self):
        return f"DecoderArchitectureConfig(kernel_sizes={self.kernel_sizes}, strides={self.strides}, " \
               f"paddings={self.paddings}, input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, " \
               f"output_dim={self.output_dim})"
