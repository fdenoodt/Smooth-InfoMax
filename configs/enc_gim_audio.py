import torch

from configs.config_classes import EncoderConfig, DataSetConfig, Dataset, OptionsConfig, Loss, ClassifierConfig
from encoder.architecture_config import ArchitectureConfig, ModuleConfig

ROOT_LOGS = r"C:\\sim_logs\\"

# Original dimensions given in CPC paper (Oord et al.).
kernel_sizes = [10, 8, 4, 4, 4]  # 20480 -> 128
strides = [5, 4, 2, 2, 2]
padding = [2, 2, 2, 2, 1]
max_pool_stride = None
max_pool_k_size = None
cnn_hidden_dim = 512
predict_distributions = False

# Splits each layer into a separate module
modules = ModuleConfig.get_modules_from_list(kernel_sizes, strides, padding, cnn_hidden_dim, predict_distributions)
ARCHITECTURE = ArchitectureConfig(modules=modules)

DATASET = DataSetConfig(
    dataset=Dataset.LIBRISPEECH,
    split_in_syllables=False,
    batch_size=8,
)

ENCODER_CONFIG = EncoderConfig(
    start_epoch=0,
    num_epochs=4,
    negative_samples=10,
    subsample=True,
    architecture=ARCHITECTURE,
    kld_weight=None,
    learning_rate=2e-4,  # 0.01  # 0.003 # old: 0.0001,
    decay_rate=1,
    train_w_noise=False,
    model_num='',  # For loading a specific model from a specific epoch and continue training
    dataset=DATASET
)

CLASSIFIER_CONFIG = ClassifierConfig(
    num_epochs=10,
    learning_rate=0.01,
    dataset=DATASET  # same batch size as encoder
)


def get_options(experiment_name):
    options = OptionsConfig(
        seed=2,
        validate=True,
        loss=Loss.INFONCE,
        model_type=0,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        experiment='audio',
        save_dir=experiment_name,
        log_path=f'{ROOT_LOGS}/{experiment_name}',
        log_every_x_epochs=1,
        model_path=f'{ROOT_LOGS}/{experiment_name}/',

        encoder_config=ENCODER_CONFIG,
        classifier_config=CLASSIFIER_CONFIG
    )

    return options


if __name__ == '__main__':
    print(f"Cuda is available: {torch.cuda.is_available()}")
