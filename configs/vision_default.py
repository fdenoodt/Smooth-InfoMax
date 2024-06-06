import torch
from config_code.config_classes import Loss, DataSetConfig, Dataset, EncoderConfig, OptionsConfig, ClassifierConfig, \
    DecoderConfig, DecoderLoss
from config_code.architecture_config import ArchitectureConfig, ModuleConfig, VisionArchitectureConfig, \
    VisionDecoderArchitectureConfig

ARCHITECTURE = VisionArchitectureConfig(
    predict_distributions=True,
    model_splits=3,
    train_module=3,
    resnet_type=50,  # 34 or 50
)

DATASET = DataSetConfig(
    dataset=Dataset.STL10,
    batch_size=16,
    grayscale=True,
    num_workers=1,  # overwrite with 16 on the cluster
)

ENCODER_CONFIG = EncoderConfig(
    start_epoch=0,
    num_epochs=300,
    negative_samples=16,
    subsample=True,
    architecture=ARCHITECTURE,
    kld_weight=0.01,
    learning_rate=1.5e-4,
    decay_rate=1,
    train_w_noise=False,
    dataset=DATASET
)

DECODER_CONFIG = DecoderConfig(
    dataset=DATASET,
    decoder_loss=DecoderLoss.MSE,
    learning_rate=1e-4,
    num_epochs=100,
    architecture=VisionDecoderArchitectureConfig(),
    encoder_num=ENCODER_CONFIG.num_epochs - 1,
)


def _get_options(experiment_name):
    options = OptionsConfig(
        config_file=__file__,
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
        vision_classifier_config=ClassifierConfig(
            learning_rate=0.01,
            dataset=DATASET,
            encoder_num=ENCODER_CONFIG.num_epochs - 1,
            num_epochs=20,
            bias=True,
        ),
        decoder_config=DECODER_CONFIG,
        use_wandb=True,
        train=True,
    )
    return options


if __name__ == '__main__':
    print(f"Cuda is available: {torch.cuda.is_available()}")
