from config_code.config_classes import EncoderConfig, DataSetConfig, Dataset, OptionsConfig, Loss, ClassifierConfig, \
    DecoderConfig, DecoderLoss
from config_code.architecture_config import ArchitectureConfig, ModuleConfig, DecoderArchitectureConfig
import torch


class SIMSetup:
    def __init__(self, predict_distributions: bool, dataset: Dataset, config_file: str, is_cpc: bool):
        self.config_file = config_file

        # Original dimensions given in CPC paper (Oord et al.).
        kernel_sizes = [10, 8, 4, 4, 4]
        strides = [5, 4, 2, 2, 2]
        padding = [2, 2, 2, 2, 1]
        max_pool_stride = None
        max_pool_k_size = None
        cnn_hidden_dim = 512  # 32
        regressor_hidden_dim = 256  # 16

        if is_cpc:
            # A single module
            modules = [
                ModuleConfig(
                    max_pool_k_size=max_pool_k_size,
                    max_pool_stride=max_pool_stride,
                    kernel_sizes=kernel_sizes,
                    strides=strides,
                    padding=padding,
                    cnn_hidden_dim=cnn_hidden_dim,
                    is_autoregressor=True,
                    regressor_hidden_dim=regressor_hidden_dim,
                    prediction_step=12,
                    predict_distributions=False,
                    is_cnn_and_autoregressor=True
                )]
        else:
            # Create three modules, one module contains two layers (except the last module: only one layer)
            modules = [
                # Three CNN modules
                ModuleConfig(
                    max_pool_k_size=max_pool_k_size,
                    max_pool_stride=max_pool_stride,
                    kernel_sizes=kernel_sizes[:2],
                    strides=strides[:2],
                    padding=padding[:2],
                    cnn_hidden_dim=cnn_hidden_dim,
                    is_autoregressor=False,
                    regressor_hidden_dim=regressor_hidden_dim,
                    prediction_step=12,
                    predict_distributions=predict_distributions
                ), ModuleConfig(
                    max_pool_k_size=max_pool_k_size,
                    max_pool_stride=max_pool_stride,
                    kernel_sizes=kernel_sizes[2:4],
                    strides=strides[2:4],
                    padding=padding[2:4],
                    cnn_hidden_dim=cnn_hidden_dim,
                    is_autoregressor=False,
                    regressor_hidden_dim=regressor_hidden_dim,
                    prediction_step=12,
                    predict_distributions=predict_distributions
                ), ModuleConfig(
                    max_pool_k_size=None,
                    max_pool_stride=None,
                    kernel_sizes=kernel_sizes[4:],
                    strides=strides[4:],
                    padding=padding[4:],
                    cnn_hidden_dim=cnn_hidden_dim,
                    is_autoregressor=False,
                    regressor_hidden_dim=regressor_hidden_dim,
                    prediction_step=12,
                    predict_distributions=predict_distributions
                ),
                # One autoregressor module
                ModuleConfig(
                    # not applicable for the regressor
                    max_pool_k_size=None, max_pool_stride=None, kernel_sizes=[], strides=[], padding=[],
                    predict_distributions=False,
                    cnn_hidden_dim=cnn_hidden_dim,
                    is_autoregressor=True,
                    regressor_hidden_dim=regressor_hidden_dim,
                    prediction_step=12,
                )]

        ARCHITECTURE = ArchitectureConfig(modules=modules, is_cpc=is_cpc)

        DATASET = DataSetConfig(
            dataset=dataset,
            split_in_syllables=False,
            batch_size=64,
            limit_train_batches=1.0,
            limit_validation_batches=1.0,
        )

        self.ENCODER_CONFIG = EncoderConfig(
            start_epoch=0,
            num_epochs=1_000,
            negative_samples=10,
            subsample=True,
            architecture=ARCHITECTURE,
            kld_weight=0.01 if predict_distributions else 0,
            learning_rate=2e-4,  # = 0.0002
            decay_rate=1,
            train_w_noise=False,
            dataset=DATASET,
        )

        self.CLASSIFIER_CONFIG_PHONES = ClassifierConfig(
            num_epochs=20,
            learning_rate=1e-4,  # = 0.0001
            # Deep copy of the dataset, to avoid changing the original dataset
            dataset=DATASET.__copy__(),
            # For loading a specific model from a specific epoch, to use by the classifier
            encoder_num=self.ENCODER_CONFIG.num_epochs - 1
        )

        self.CLASSIFIER_CONFIG_SPEAKERS = ClassifierConfig(
            num_epochs=50,
            learning_rate=1e-3,  # = 0.001
            dataset=DATASET.__copy__(),
            # For loading a specific model from a specific epoch, to use by the classifier
            encoder_num=self.ENCODER_CONFIG.num_epochs - 1
        )

        self.CLASSIFIER_CONFIG_SYLLABLES = ClassifierConfig(
            num_epochs=50,
            learning_rate=1e-3,  # = 0.001
            dataset=DataSetConfig(
                dataset=dataset,
                split_in_syllables=True,
                labels="syllables",
                batch_size=64,
                limit_train_batches=1.0,
                limit_validation_batches=1.0,
            ),
            # For loading a specific model from a specific epoch, to use by the classifier
            encoder_num=self.ENCODER_CONFIG.num_epochs - 1
        )

        self.DECODER_CONFIG = DecoderConfig(
            num_epochs=200,
            learning_rate=2e-4,
            dataset=DataSetConfig(
                dataset=dataset,
                split_in_syllables=False,
                batch_size=64,
                limit_train_batches=1.0,
                limit_validation_batches=1.0,
            ),
            encoder_num=self.ENCODER_CONFIG.num_epochs - 1,
            architecture=DecoderArchitectureConfig(
                # decoder is trained on second to last module (skip the autoregressor)
                kernel_sizes=kernel_sizes[::-1],
                strides=strides[::-1],
                paddings=padding[::-1],
                output_paddings=[1, 0, 1, 3, 4],
                input_dim=cnn_hidden_dim,
                hidden_dim=cnn_hidden_dim,
                output_dim=1
            ),
            decoder_loss=DecoderLoss.MSE_MEL
        )

    def get_options(self, experiment_name) -> OptionsConfig:
        options = OptionsConfig(
            config_file=self.config_file,
            seed=-1,
            validate=True,
            loss=Loss.INFO_NCE,
            experiment='audio',
            save_dir=experiment_name,
            log_every_x_epochs=1,
            encoder_config=self.ENCODER_CONFIG,
            phones_classifier_config=self.CLASSIFIER_CONFIG_PHONES,
            speakers_classifier_config=self.CLASSIFIER_CONFIG_SPEAKERS,
            syllables_classifier_config=self.CLASSIFIER_CONFIG_SYLLABLES,
            decoder_config=self.DECODER_CONFIG,
            vision_classifier_config=None
        )

        return options
