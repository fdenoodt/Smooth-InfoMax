import torch

from config_code.config_classes import Loss, DataSetConfig, Dataset, EncoderConfig, OptionsConfig
from config_code.architecture_config import ArchitectureConfig, ModuleConfig

# WARNING: CURRENT BUG: THIS NAME SHOULD BE THE SAME AS WHERE CPC LOCATION,
# if not: inconsitent results
# (see options_decoder.py > `gim_model_path`)

NUM_EPOCHS = 4
START_EPOCH = 0
BATCH_SIZE = 8  # 171

# Simple architecture v2 # 20480 -> 105 (first module)
kernel_sizes = [10, 8, 3]
strides = [4, 3, 1]
padding = [2, 2, 1]
max_pool_k_size = 8
max_pool_stride = 4
cnn_hidden_dim = 32
regressor_hidden_dim = 16
predict_distributions = True

module1 = ModuleConfig(
    max_pool_k_size=max_pool_k_size, max_pool_stride=max_pool_stride,
    kernel_sizes=kernel_sizes, strides=strides, padding=padding,
    cnn_hidden_dim=cnn_hidden_dim, regressor_hidden_dim=regressor_hidden_dim, is_autoregressor=False,
    prediction_step=12, predict_distributions=predict_distributions)

# Second module architectures: (v2 / v3)
# v2
# kernel_sizes = [8, 8, 3]
# strides = [3, 3, 1]
# padding = [2, 2, 1]
# max_pool_k_size = None
# max_pool_stride = None

# v3
kernel_sizes = [6, 6, 3]
strides = [2, 2, 1]
padding = [2, 2, 1]
max_pool_k_size = None
max_pool_stride = None
module2 = ModuleConfig(
    max_pool_k_size=None, max_pool_stride=None,
    kernel_sizes=kernel_sizes, strides=strides, padding=padding,
    cnn_hidden_dim=cnn_hidden_dim, regressor_hidden_dim=regressor_hidden_dim, is_autoregressor=False,
    prediction_step=12, predict_distributions=True)

ARCHITECTURE = ArchitectureConfig(modules=[module1, module2])

DATASET = DataSetConfig(
    dataset=Dataset.xxxx,
    split_in_syllables=False,
    batch_size=8,
)

ENCODER_CONFIG = EncoderConfig(
    start_epoch=0,
    num_epochs=4,
    negative_samples=10,
    subsample=True,
    architecture=ARCHITECTURE,
    kld_weight=0.0033,
    learning_rate=2e-4,  # 0.01  # 0.003 # old: 0.0001,
    decay_rate=0.99,
    train_w_noise=False,
    dataset=DATASET
)


def _get_options(experiment_name):
    options = OptionsConfig(
        seed=2,
        validate=True,
        loss=Loss.INFO_NCE,
        encoder_config=ENCODER_CONFIG,
        experiment='audio',
        save_dir=experiment_name,
        log_every_x_epochs=1,
        phones_classifier_config=None,
        speakers_classifier_config=None,
        syllables_classifier_config=None,
        decoder_config=None,
    )
    return options


if __name__ == '__main__':
    print(f"Cuda is available: {torch.cuda.is_available()}")
