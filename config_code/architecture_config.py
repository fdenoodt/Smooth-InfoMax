from typing import Optional, List, Union


class ModuleConfig:
    def __init__(self,
                 max_pool_k_size: Optional[int], max_pool_stride: Optional[int], kernel_sizes: list,
                 is_autoregressor: bool, prediction_step: int, predict_distributions: bool,
                 strides: list, padding: list, non_linearities: list, cnn_hidden_dim: int,
                 regressor_hidden_dim: Optional[int],
                 is_cnn_and_autoregressor: Optional[bool] = False):  # for CPC
        assert len(kernel_sizes) == len(strides) == len(padding)

        if is_autoregressor and not (is_cnn_and_autoregressor):
            # when a typical regressor module from Sindy (then no cnn layer)
            assert len(kernel_sizes) == 0
            assert len(strides) == 0
            assert len(padding) == 0
            assert len(non_linearities) == 0

            assert max_pool_k_size is None
            assert max_pool_stride is None

        self.max_pool_k_size = max_pool_k_size
        self.max_pool_stride = max_pool_stride
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.padding = padding

        # Typically in GIM+SIM always true. but at some layers we set them False for CPC in the density experiments.
        # That way same number of layers are used in all experiments, and can be compared more easily.
        # For performance experiments, we set them to True.
        self.non_linearities = non_linearities

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


class ArchitectureConfig:  # for encoder, only AUDIO
    def __init__(self, modules: list[ModuleConfig], use_batch_norm: bool, is_cpc: Optional[bool] = False):
        if is_cpc:
            assert len(modules) == 1

        self.is_cpc = is_cpc
        self.use_batch_norm = use_batch_norm
        self.modules: list[ModuleConfig] = modules

    def __str__(self):
        modules: str = ", ".join([str(module) for module in self.modules])
        return f"ArchitectureConfig(modules=[{modules}], use_batch_norm={self.use_batch_norm}, is_cpc={self.is_cpc})"


class VisionArchitectureConfig:
    def __init__(self, predict_distributions: bool, model_splits: int, train_module: int, resnet_type: int):
        self.predict_distributions = predict_distributions
        self.model_splits = model_splits
        self.train_module = train_module
        self.modules: List[int] = [0] * model_splits  # [0, 0, 0, 0, 0], dummy variable needed in `logger.py`

        self._resnet_type = None
        self.hidden_dim = None
        self.resnet_type = resnet_type

    @property
    def resnet_type(self):
        return self._resnet_type

    @resnet_type.setter
    def resnet_type(self, value):
        # working with a setter such that if after initialization, the value is changed, the hidden_dim is also updated
        # this is relevant when overriding the config!
        assert value in [50, 34], "resnet_type must be 50 or 34"
        self._resnet_type = value
        if self._resnet_type == 50:
            self.hidden_dim = 1024
        else:
            self.hidden_dim = 256

    def __str__(self):
        return (f"VisionArchitectureConfig(predict_distributions={self.predict_distributions}, "
                f"model_splits={self.model_splits}, "
                f"train_module={self.train_module}, resnet_type={self.resnet_type})")


class DecoderArchitectureConfig:
    def __init__(self, kernel_sizes: List[int], strides: List[int], paddings: List[int], output_paddings: List[int],
                 input_dim: int, hidden_dim: int, output_dim: int, expected_nb_frames_latent_repr: int):
        assert len(kernel_sizes) == len(strides) == len(paddings) == len(output_paddings)
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.paddings = paddings
        self.output_paddings = output_paddings
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.expected_nb_frames_latent_repr: int = expected_nb_frames_latent_repr  # eg 64, used for Gaussian Sampling

    def __str__(self):
        return (f"DecoderArchitectureConfig(kernel_sizes={self.kernel_sizes}, strides={self.strides}, "
                f"paddings={self.paddings}, output_paddings={self.output_paddings}, input_dim={self.input_dim}, "
                f"hidden_dim={self.hidden_dim}, output_dim={self.output_dim})")


class VisionDecoderArchitectureConfig:
    def __init__(self):
        pass

    def __str__(self):
        return "VisionDecoderArchitectureConfig()"
