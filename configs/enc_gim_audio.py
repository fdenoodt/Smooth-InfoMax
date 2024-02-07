import torch

from configs.config_classes import EncoderConfig, DataSetConfig, Dataset, OptionsConfig, Loss, ClassifierConfig, \
    ModelType
from encoder.architecture_config import ArchitectureConfig, ModuleConfig

ROOT_LOGS = r"./sim_logs/"

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
modules = ModuleConfig.get_modules_from_list(kernel_sizes, strides, padding, cnn_hidden_dim, predict_distributions)
modules.append(
    ModuleConfig(
        # not applicable for the regressor
        max_pool_k_size=None, max_pool_stride=None, kernel_sizes=[], strides=[], padding=[],

        cnn_hidden_dim=cnn_hidden_dim,
        is_autoregressor=True,
        regressor_hidden_dim=regressor_hidden_dim,
        prediction_step=12,
        predict_distributions=predict_distributions
    )
)
ARCHITECTURE = ArchitectureConfig(modules=modules)

DATASET = DataSetConfig(
    dataset=Dataset.LIBRISPEECH_SUBSET,
    split_in_syllables=False,
    batch_size=8,
)

ENCODER_CONFIG = EncoderConfig(
    start_epoch=0,
    num_epochs=1_000,
    negative_samples=10,
    subsample=True,
    architecture=ARCHITECTURE,
    kld_weight=None,
    learning_rate=2e-4,  # = 0.0002
    decay_rate=1,
    train_w_noise=False,
    dataset=DATASET,
)

CLASSIFIER_CONFIG_PHONES = ClassifierConfig(
    num_epochs=20,
    learning_rate=1e-4,  # = 0.0001
    # Deep copy of the dataset, to avoid changing the original dataset
    dataset=DATASET.__copy__(),
    # For loading a specific model from a specific epoch, to use by the classifier
    encoder_num=ENCODER_CONFIG.num_epochs - 1
)
CLASSIFIER_CONFIG_PHONES.dataset.batch_size = 8

CLASSIFIER_CONFIG_SPEAKERS = ClassifierConfig(
    num_epochs=50,
    learning_rate=1e-3,  # = 0.001
    dataset=DATASET.__copy__(),
    # For loading a specific model from a specific epoch, to use by the classifier
    encoder_num=ENCODER_CONFIG.num_epochs - 1
)
CLASSIFIER_CONFIG_SPEAKERS.dataset.batch_size = 64


def get_options(experiment_name) -> OptionsConfig:
    options = OptionsConfig(
        seed=2,
        validate=True,
        loss=Loss.INFO_NCE,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        experiment='audio',
        save_dir=experiment_name,
        log_path=f'{ROOT_LOGS}/{experiment_name}',
        log_every_x_epochs=1,
        model_path=f'{ROOT_LOGS}/{experiment_name}/',

        encoder_config=ENCODER_CONFIG,
        phones_classifier_config=CLASSIFIER_CONFIG_PHONES,
        speakers_classifier_config=CLASSIFIER_CONFIG_SPEAKERS
    )

    return options


if __name__ == '__main__':
    print(f"Cuda is available: {torch.cuda.is_available()}")
