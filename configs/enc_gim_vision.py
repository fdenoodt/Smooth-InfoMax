import torch

from config_code.config_classes import EncoderConfig, DataSetConfig, Dataset, OptionsConfig, Loss, ClassifierConfig
from config_code.architecture_config import ArchitectureConfig, ModuleConfig

# Original dimensions given in CPC paper (Oord et al.).
kernel_sizes = [10, 8, 4, 4, 4]
strides = [5, 4, 2, 2, 2]
padding = [2, 2, 2, 2, 1]
max_pool_stride = None
max_pool_k_size = None
cnn_hidden_dim = 512
regressor_hidden_dim = 256
predict_distributions = False

# Splits each layer into a separate module
# modules = ModuleConfig.get_modules_from_list(kernel_sizes, strides, padding, cnn_hidden_dim, predict_distributions)
# modules.append(
#     ModuleConfig(
#         # not applicable for the regressor
#         max_pool_k_size=None, max_pool_stride=None, kernel_sizes=[], strides=[], padding=[],
#
#         cnn_hidden_dim=cnn_hidden_dim,
#         is_autoregressor=True,
#         regressor_hidden_dim=regressor_hidden_dim,
#         prediction_step=5,
#         predict_distributions=predict_distributions
#     )
# )
ARCHITECTURE = ArchitectureConfig(modules=[])

DATASET = DataSetConfig(
    dataset=Dataset.STL10, grayscale=True,
    batch_size=32,
    limit_train_batches=1.0,
    limit_validation_batches=1.0)

ENCODER_CONFIG = EncoderConfig(
    start_epoch=0,
    num_epochs=300,
    negative_samples=16,
    subsample=True,
    architecture=ARCHITECTURE,
    kld_weight=None,
    learning_rate=0.0002,  # = 2e-4
    decay_rate=1,
    train_w_noise=False,
    dataset=DATASET,
)


def _get_options(experiment_name) -> OptionsConfig:
    options = OptionsConfig(
        seed=2,
        validate=True,
        loss=Loss.INFO_NCE,
        experiment='vision',
        save_dir=experiment_name,
        log_every_x_epochs=1,
        encoder_config=ENCODER_CONFIG,
        phones_classifier_config=None,
        speakers_classifier_config=None
    )

    return options


if __name__ == '__main__':
    print(f"Cuda is available: {torch.cuda.is_available()}")
