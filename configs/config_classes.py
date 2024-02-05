from enum import Enum
from typing import Optional
import os

from encoder.architecture_config import ArchitectureConfig


class Loss(Enum):
    # InfoNCE loss, supervised loss using the phone labels, supervised loss using the phone labels
    INFONCE = 0
    SUPERVISED_PHONES = 1
    SUPERVISED_SPEAKER = 2


class Dataset(Enum):
    # de_boer_sounds OR librispeech OR de_boer_sounds_reshuffled
    LIBRISPEECH = 1
    DE_BOER = 2
    DE_BOER_RESHUFFLED = 3
    DE_BOER_RESHUFFLED_V2 = 4


class DataSetConfig:
    def __init__(self, dataset: Dataset, split_in_syllables, batch_size, labels: Optional[str] = None):
        self.data_input_dir = './datasets/'
        self.dataset: Dataset = dataset
        self.split_in_syllables = split_in_syllables
        self.batch_size = batch_size
        self.batch_size_multiGPU = batch_size  # will be overwritten in model_utils.distribute_over_GPUs
        self.labels = labels  # eg: syllables or vowels, only for de_boer_sounds dataset

        if split_in_syllables:
            assert dataset in [Dataset.DE_BOER, Dataset.DE_BOER_RESHUFFLED]
            "split_in_syllables can only be True for de_boer_sounds dataset"


class EncoderConfig:
    def __init__(self, start_epoch, num_epochs, negative_samples, subsample, architecture: ArchitectureConfig,
                 kld_weight, learning_rate, decay_rate,
                 train_w_noise, model_num, dataset: DataSetConfig):
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs
        self.negative_samples = negative_samples
        self.subsample = subsample
        self.architecture: ArchitectureConfig = architecture
        self.kld_weight = kld_weight
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.train_w_noise = train_w_noise
        self.model_num = model_num
        self.dataset = dataset


class ClassifierConfig:
    def __init__(self, num_epochs, learning_rate, dataset: DataSetConfig):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.dataset = dataset


class OptionsConfig:
    def __init__(self,
                 seed, validate, loss: Loss, encoder_config, model_type, device, experiment, save_dir, log_path,
                 log_every_x_epochs, model_path, classifier_config: Optional[ClassifierConfig]):
        self.seed = seed
        self.validate = validate
        self.loss = loss
        self.model_type = model_type
        self.device = device
        self.experiment = experiment
        self.save_dir = save_dir
        self.log_path = log_path
        self.log_path_latent = os.path.join(log_path, "latent_space")

        self.log_every_x_epochs = log_every_x_epochs
        self.model_path = model_path

        self.encoder_config: EncoderConfig = encoder_config
        self.classifier_config: ClassifierConfig = classifier_config
