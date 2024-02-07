from typing import Optional


class ModuleConfig:
    def __init__(self,
                 max_pool_k_size: Optional[int], max_pool_stride: Optional[int], kernel_sizes: list,
                 strides: list, padding: list, cnn_hidden_dim: int, regressor_hidden_dim: Optional[int],
                 is_autoregressor: bool, prediction_step: int, predict_distributions: bool):

        assert len(kernel_sizes) == len(strides) == len(padding)
        if is_autoregressor:
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


class ArchitectureConfig:
    def __init__(self, modules: list):
        self.modules = modules
