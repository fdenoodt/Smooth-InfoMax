from config_code.config_classes import EncoderConfig, DataSetConfig, Dataset, OptionsConfig, Loss, ClassifierConfig
from config_code.architecture_config import ArchitectureConfig, ModuleConfig
import torch

class SIMSetup:
    def __init__(self, predict_distributions: bool):
        # Original dimensions given in CPC paper (Oord et al.).
        kernel_sizes = [10, 8, 4, 4, 4]
        strides = [5, 4, 2, 2, 2]
        padding = [2, 2, 2, 2, 1]
        max_pool_stride = None
        max_pool_k_size = None
        cnn_hidden_dim = 512
        regressor_hidden_dim = 256

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

        ARCHITECTURE = ArchitectureConfig(modules=modules)

        DATASET = DataSetConfig(
            dataset=Dataset.LIBRISPEECH,
            split_in_syllables=False,
            batch_size=8,
            limit_train_batches=1.0,
            limit_validation_batches=1.0,
        )

        self.ENCODER_CONFIG = EncoderConfig(
            start_epoch=0,
            num_epochs=1_000,
            negative_samples=10,
            subsample=True,
            architecture=ARCHITECTURE,
            kld_weight=0.1,
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
        self.CLASSIFIER_CONFIG_PHONES.dataset.batch_size = 8

        self.CLASSIFIER_CONFIG_SPEAKERS = ClassifierConfig(
            num_epochs=50,
            learning_rate=1e-3,  # = 0.001
            dataset=DATASET.__copy__(),
            # For loading a specific model from a specific epoch, to use by the classifier
            encoder_num=self.ENCODER_CONFIG.num_epochs - 1
        )
        self.CLASSIFIER_CONFIG_SPEAKERS.dataset.batch_size = 64

    def get_options(self, experiment_name) -> OptionsConfig:
        options = OptionsConfig(
            seed=2,
            validate=True,
            loss=Loss.INFO_NCE,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            experiment='audio',
            save_dir=experiment_name,
            log_every_x_epochs=1,
            encoder_config=self.ENCODER_CONFIG,
            phones_classifier_config=self.CLASSIFIER_CONFIG_PHONES,
            speakers_classifier_config=self.CLASSIFIER_CONFIG_SPEAKERS
        )

        return options
