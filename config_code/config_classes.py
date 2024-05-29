from enum import Enum
from typing import Optional, Union
import os
import torch
import datetime

from config_code.architecture_config import ArchitectureConfig, DecoderArchitectureConfig, VisionArchitectureConfig


class Loss(Enum):
    # InfoNCE loss, supervised loss using the phone labels, supervised loss using the phone labels
    INFO_NCE = 0
    SUPERVISED_PHONES = 1
    SUPERVISED_SPEAKER = 2
    SUPERVISED_VISUAL = 3


class Dataset(Enum):
    # de_boer_sounds OR librispeech OR de_boer_sounds_reshuffled
    LIBRISPEECH = 1
    LIBRISPEECH_SUBSET = 3
    DE_BOER = 4  # used to be 5, i think irrelevant for classification
    # DE_BOER_RESHUFFLED_V2 = 6
    STL10 = 7
    ANIMAL_WITH_ATTRIBUTES = 8


class ModelType(Enum):
    UNDEFINED = 0
    FULLY_SUPERVISED = 1  # Both the downstream and the encoder are trained
    ONLY_DOWNSTREAM_TASK = 2  # Only the downstream task, encoder is frozen
    ONLY_ENCODER = 3  # Only the encoder is trained


class DataSetConfig:
    def __init__(self, dataset: Dataset, batch_size, labels: Optional[str] = None,
                 limit_train_batches: Optional[float] = 1.0, limit_validation_batches: Optional[float] = 1.0,
                 grayscale: Optional[bool] = False, split_in_syllables: Optional[bool] = False,
                 num_workers: Optional[int] = 0):
        self.data_input_dir = './datasets/'
        self.dataset: Dataset = dataset
        self.split_in_syllables = split_in_syllables
        self.batch_size = batch_size
        self.batch_size_multiGPU = batch_size  # will be overwritten in model_utils.distribute_over_GPUs
        self.num_workers = num_workers

        if split_in_syllables:
            assert dataset in [Dataset.DE_BOER]
            "split_in_syllables can only be True for de_boer_sounds dataset"

        if (split_in_syllables and dataset in [Dataset.DE_BOER]):
            assert labels in ["syllables", "vowels"]

        if grayscale:
            assert dataset in [Dataset.STL10, Dataset.ANIMAL_WITH_ATTRIBUTES]
            "grayscale can only be True for STL10 dataset or ANIMAL_WITH_ATTRIBUTES dataset"

        self.labels = labels  # eg: syllables or vowels, only for de_boer_sounds dataset
        self.limit_train_batches = limit_train_batches
        self.limit_validation_batches = limit_validation_batches
        self.grayscale = grayscale

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
    def __init__(self, start_epoch, num_epochs, negative_samples, subsample,
                 architecture: Union[ArchitectureConfig, VisionArchitectureConfig],
                 kld_weight, learning_rate, decay_rate,
                 train_w_noise, dataset: DataSetConfig,
                 deterministic: Optional[bool] = False):
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.negative_samples = negative_samples
        self.subsample = subsample
        self.architecture: Union[ArchitectureConfig, VisionArchitectureConfig] = architecture
        self.kld_weight = kld_weight
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.train_w_noise = train_w_noise
        self.dataset = dataset

        # Useful after training to get deterministic results. If True, the encoder will use mode of the posterior distribution
        self.deterministic = deterministic

    def __str__(self):
        return f"EncoderConfig(start_epoch={self.start_epoch}, num_epochs={self.num_epochs}, " \
               f"negative_samples={self.negative_samples}, subsample={self.subsample}, " \
               f"architecture={self.architecture}, kld_weight={self.kld_weight}, " \
               f"learning_rate={self.learning_rate}, decay_rate={self.decay_rate}, " \
               f"train_w_noise={self.train_w_noise}, dataset={self.dataset})"


class ClassifierConfig:
    def __init__(self, num_epochs, learning_rate, dataset: DataSetConfig, encoder_num: str,
                 bias: Optional[bool] = True):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.encoder_num = encoder_num
        self.bias = bias

    # to string
    def __str__(self):
        return f"ClassifierConfig(num_epochs={self.num_epochs}, learning_rate={self.learning_rate}, " \
               f"dataset={self.dataset}, encoder_num={self.encoder_num})"


class DecoderLoss(Enum):
    MSE = 0
    SPECTRAL = 1
    MSE_SPECTRAL = 2
    FFT = 3
    MSE_FFT = 4
    MEL = 5
    MSE_MEL = 6


class DecoderConfig:
    def __init__(self, num_epochs, learning_rate, dataset: DataSetConfig, encoder_num: str,
                 architecture: DecoderArchitectureConfig, decoder_loss: DecoderLoss):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.dataset = dataset
        self.encoder_num = encoder_num
        self.architecture: DecoderArchitectureConfig = architecture
        self.decoder_loss: DecoderLoss = decoder_loss


class OptionsConfig:
    def __init__(self, config_file, seed, validate, loss: Loss, encoder_config, experiment,
                 save_dir,
                 log_every_x_epochs, phones_classifier_config: Optional[ClassifierConfig],
                 speakers_classifier_config: Optional[ClassifierConfig],
                 syllables_classifier_config: Optional[ClassifierConfig],
                 decoder_config: Optional[DecoderConfig],
                 vision_classifier_config: Optional[ClassifierConfig],
                 use_wandb: Optional[bool] = True
                 ):
        root_logs = r"./sim_logs/"

        # current time
        self.time = datetime.datetime.now()
        self.model_type: ModelType = ModelType.UNDEFINED  # will be set in the main function

        self.config_file = config_file
        self.seed = seed
        self.validate = validate
        self.loss = loss
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.experiment = experiment
        self.save_dir = save_dir
        self.log_path = f'{root_logs}/{save_dir}'
        self.log_path_latent = os.path.join(f'{root_logs}/{save_dir}', "latent_space")

        self.log_every_x_epochs = log_every_x_epochs
        self.model_path = f'{root_logs}/{save_dir}'

        self.encoder_config: EncoderConfig = encoder_config
        self.phones_classifier_config: Optional[ClassifierConfig] = phones_classifier_config
        self.speakers_classifier_config: Optional[ClassifierConfig] = speakers_classifier_config
        self.syllables_classifier_config: Optional[ClassifierConfig] = syllables_classifier_config
        self.decoder_config: Optional[DecoderConfig] = decoder_config

        self.vision_classifier_config: Optional[ClassifierConfig] = vision_classifier_config
        self.use_wandb = use_wandb

    def __str__(self):
        return f"OptionsConfig(model_type={self.model_type}, seed={self.seed}, validate={self.validate}, " \
               f"loss={self.loss}, encoder_config={self.encoder_config}, device={self.device}, " \
               f"experiment={self.experiment}, save_dir={self.save_dir}, log_path={self.log_path}, " \
               f"log_every_x_epochs={self.log_every_x_epochs}, model_path={self.model_path}, " \
               f"phones_classifier_config={self.phones_classifier_config}, speakers_classifier_config={self.speakers_classifier_config})"
