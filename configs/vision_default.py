import torch
from config_code.config_classes import Loss, DataSetConfig, Dataset, EncoderConfig, OptionsConfig
from config_code.architecture_config import ArchitectureConfig, ModuleConfig

ARCHITECTURE = ArchitectureConfig(modules=[1, 2, 3])

DATASET = DataSetConfig(
    dataset=Dataset.STL10,
    batch_size=8,
    grayscale=True,
)

ENCODER_CONFIG = EncoderConfig(
    start_epoch=0,
    num_epochs=4,
    negative_samples=16,
    subsample=True,
    architecture=ARCHITECTURE,
    kld_weight=0.0033,
    learning_rate=2e-4,  # 0.01  # 0.003 # old: 0.0001,
    decay_rate=1,
    train_w_noise=False,
    dataset=DATASET
)


def _get_options(experiment_name):
    options = OptionsConfig(
        seed=2,
        validate=True,
        loss=Loss.INFO_NCE,
        encoder_config=ENCODER_CONFIG,
        experiment='vision',
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
