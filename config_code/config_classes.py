from enum import Enum
from typing import Optional
import os
import torch

from config_code.architecture_config import ArchitectureConfig, DecoderArchitectureConfig


class Loss(Enum):
    # InfoNCE loss, supervised loss using the phone labels, supervised loss using the phone labels
    INFO_NCE = 0
    SUPERVISED_PHONES = 1
    SUPERVISED_SPEAKER = 2


class Dataset(Enum):
    # de_boer_sounds OR librispeech OR de_boer_sounds_reshuffled
    LIBRISPEECH = 1
    LIBRISPEECH_SUBSET = 3
    DE_BOER = 4
    DE_BOER_RESHUFFLED = 5
    DE_BOER_RESHUFFLED_V2 = 6


class ModelType(Enum):
    UNDEFINED = 0
    FULLY_SUPERVISED = 1  # Both the downstream and the encoder are trained
    ONLY_DOWNSTREAM_TASK = 2  # Only the downstream task, encoder is frozen
    ONLY_ENCODER = 3  # Only the encoder is trained


class DataSetConfig:
    def __init__(self, dataset: Dataset, split_in_syllables, batch_size, labels: Optional[str] = None,
                 limit_train_batches: Optional[float] = 1.0, limit_validation_batches: Optional[float] = 1.0):
        self.data_input_dir = './datasets/'
        self.dataset: Dataset = dataset
        self.split_in_syllables = split_in_syllables
        self.batch_size = batch_size
        self.batch_size_multiGPU = batch_size  # will be overwritten in model_utils.distribute_over_GPUs
        self.labels = labels  # eg: syllables or vowels, only for de_boer_sounds dataset
        self.limit_train_batches = limit_train_batches
        self.limit_validation_batches = limit_validation_batches

        if split_in_syllables:
            assert dataset in [Dataset.DE_BOER, Dataset.DE_BOER_RESHUFFLED]
            "split_in_syllables can only be True for de_boer_sounds dataset"

    def __copy__(self):
        return DataSetConfig(
            dataset=self.dataset,
            split_in_syllables=self.split_in_syllables,
            batch_size=self.batch_size,
            labels=self.labels
        )

    def __str__(self):
        return f"DataSetConfig(dataset={self.dataset}, split_in_syllables={self.split_in_syllables}, " \
               f"batch_size={self.batch_size}, labels={self.labels})"


class EncoderConfig:
    def __init__(self, start_epoch, num_epochs, negative_samples, subsample, architecture: ArchitectureConfig,
                 kld_weight, learning_rate, decay_rate,
                 train_w_noise, dataset: DataSetConfig):
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.negative_samples = negative_samples
        self.subsample = subsample
        self.architecture: ArchitectureConfig = architecture
        self.kld_weight = kld_weight
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.train_w_noise = train_w_noise
        self.dataset = dataset

    def __str__(self):
        return f"EncoderConfig(start_epoch={self.start_epoch}, num_epochs={self.num_epochs}, " \
               f"negative_samples={self.negative_samples}, subsample={self.subsample}, " \
               f"architecture={self.architecture}, kld_weight={self.kld_weight}, " \
               f"learning_rate={self.learning_rate}, decay_rate={self.decay_rate}, " \
               f"train_w_noise={self.train_w_noise}, dataset={self.dataset})"


class ClassifierConfig:
    def __init__(self, num_epochs, learning_rate, dataset: DataSetConfig, encoder_num: str):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.encoder_num = encoder_num

    # to string
    def __str__(self):
        return f"ClassifierConfig(num_epochs={self.num_epochs}, learning_rate={self.learning_rate}, " \
               f"dataset={self.dataset}, encoder_num={self.encoder_num})"


class DecoderConfig:
    def __init__(self, num_epochs, learning_rate, dataset: DataSetConfig, encoder_num: str,
                 architecture: DecoderArchitectureConfig):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.encoder_num = encoder_num
        self.architecture: DecoderArchitectureConfig = architecture


class OptionsConfig:
    def __init__(self, seed, validate, loss: Loss, encoder_config, experiment,
                 save_dir,
                 log_every_x_epochs, phones_classifier_config: Optional[ClassifierConfig],
                 speakers_classifier_config: Optional[ClassifierConfig],
                 syllables_classifier_config: Optional[ClassifierConfig],
                 decoder_config: Optional[DecoderConfig]):
        root_logs = r"./sim_logs/"

        self.model_type: ModelType = ModelType.UNDEFINED  # will be set in the main function
        self.seed = seed
        self.validate = validate
        self.loss = loss
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.experiment = experiment
        self.save_dir = save_dir
        self.log_path = f'{root_logs}/{save_dir}'
        self.log_path_latent = os.path.join(save_dir, "latent_space")

        self.log_every_x_epochs = log_every_x_epochs
        self.model_path = f'{root_logs}/{save_dir}'

        self.encoder_config: EncoderConfig = encoder_config
        self.phones_classifier_config: Optional[ClassifierConfig] = phones_classifier_config
        self.speakers_classifier_config: Optional[ClassifierConfig] = speakers_classifier_config
        self.syllables_classifier_config: Optional[ClassifierConfig] = syllables_classifier_config
        self.decoder_config: Optional[DecoderConfig] = decoder_config

    def __str__(self):
        return f"OptionsConfig(model_type={self.model_type}, seed={self.seed}, validate={self.validate}, " \
               f"loss={self.loss}, encoder_config={self.encoder_config}, device={self.device}, " \
               f"experiment={self.experiment}, save_dir={self.save_dir}, log_path={self.log_path}, " \
               f"log_every_x_epochs={self.log_every_x_epochs}, model_path={self.model_path}, " \
               f"phones_classifier_config={self.phones_classifier_config}, speakers_classifier_config={self.speakers_classifier_config})"
