from enum import Enum
from typing import Optional, Union, List, Dict, Tuple
import os
import torch
import datetime

from config_code.architecture_config import ArchitectureConfig, DecoderArchitectureConfig, VisionArchitectureConfig, \
    VisionDecoderArchitectureConfig


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
    SHAPES_3D = 9
    SHAPES_3D_SUBSET = 10  # only used for local development, not in the cluster!
    RADIO = 11


class Label(Enum):
    DEFAULT = 0  # when no ambiguity
    PHONES = 1  # (Libri)
    SPEAKERS = 2  # (Libri)
    SYLLABLES = 3  # (de boer)
    VOWELS = 4  # (de boer)


class ClassifierKey(Enum):
    LIBRI_PHONES = 1
    LIBRI_SPEAKERS = 2
    DE_BOER_SYLLABLES = 3
    DE_BOER_VOWELS = 4
    RADIO = 5


class ModelType(Enum):
    UNDEFINED = 0
    FULLY_SUPERVISED = 1  # Both the downstream and the encoder are trained
    ONLY_DOWNSTREAM_TASK = 2  # Only the downstream task, encoder is frozen
    ONLY_ENCODER = 3  # Only the encoder is trained


class DataSetConfig:
    def __init__(self, dataset: Dataset, batch_size, labels: Label,
                 limit_train_batches: Optional[float] = 1.0, limit_validation_batches: Optional[float] = 1.0,
                 grayscale: Optional[bool] = False,
                 num_workers: Optional[int] = 0):
        self.data_input_dir = './datasets/'
        self.dataset: Dataset = dataset

        # self.split_in_syllables = split_in_syllables
        self.batch_size = batch_size
        self.batch_size_multiGPU = batch_size  # will be overwritten in model_utils.distribute_over_GPUs
        self.num_workers = num_workers

        if dataset in [Dataset.LIBRISPEECH, Dataset.LIBRISPEECH_SUBSET]:
            assert labels in [Label.PHONES, Label.SPEAKERS]

        if dataset in [Dataset.DE_BOER]:
            assert labels in [Label.DEFAULT,  # no splitting into syllables/vowels
                              Label.SYLLABLES,
                              Label.VOWELS]

        if dataset == Dataset.RADIO:
            assert labels == Label.DEFAULT

        if dataset in [Dataset.STL10, Dataset.ANIMAL_WITH_ATTRIBUTES, Dataset.SHAPES_3D]:
            assert labels == Label.DEFAULT

        # if split_in_syllables:
        #     assert dataset in [Dataset.DE_BOER]
        #     "split_in_syllables can only be True for de_boer_sounds dataset"

        # if (split_in_syllables and dataset in [Dataset.DE_BOER]):
        #     assert labels in ["syllables", "vowels"]

        if grayscale:
            assert dataset in [Dataset.STL10, Dataset.ANIMAL_WITH_ATTRIBUTES, Dataset.SHAPES_3D]
            "grayscale can only be True for STL10 dataset or ANIMAL_WITH_ATTRIBUTES dataset"

        self.labels: Label = labels  # eg: syllables or vowels, only for de_boer_sounds dataset
        self.limit_train_batches = limit_train_batches
        self.limit_validation_batches = limit_validation_batches
        self.grayscale = grayscale

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        if not isinstance(value, Dataset):
            raise TypeError("dataset must be an instance of Dataset Enum")
        self._dataset = value
        # Update the dataset-dependent assertions or configurations here if needed

        # also update labels to default if dataset is not de_boer
        if value != Dataset.DE_BOER:
            self.labels = Label.DEFAULT

    def __copy__(self):
        return DataSetConfig(
            dataset=self.dataset,
            batch_size=self.batch_size,
            labels=self.labels,
            limit_train_batches=self.limit_train_batches,
            limit_validation_batches=self.limit_validation_batches,
            grayscale=self.grayscale,
            num_workers=self.num_workers
        )

    def __str__(self):
        return f"DataSetConfig(dataset={self.dataset}, batch_size={self.batch_size}, labels={self.labels}, " \
               f"limit_train_batches={self.limit_train_batches}, limit_validation_batches={self.limit_validation_batches}, " \
               f"grayscale={self.grayscale}, num_workers={self.num_workers})"


class EncoderConfig:
    def __init__(self, start_epoch, num_epochs, negative_samples, subsample,
                 architecture: Union[ArchitectureConfig, VisionArchitectureConfig],
                 kld_weight, learning_rate, decay_rate,
                 train_w_noise,
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

        # Useful after training to get deterministic results. If True, the encoder will use mode of the posterior distribution
        self.deterministic = deterministic

    def __str__(self):
        return f"EncoderConfig(start_epoch={self.start_epoch}, num_epochs={self.num_epochs}, " \
               f"negative_samples={self.negative_samples}, subsample={self.subsample}, " \
               f"architecture={self.architecture}, kld_weight={self.kld_weight}, " \
               f"learning_rate={self.learning_rate}, decay_rate={self.decay_rate}, " \
               f"train_w_noise={self.train_w_noise}, deterministic={self.deterministic})"


class PostHocModel:  # Classifier or Decoder
    """encoder_module and encoder_layer are currently only supported for the audio encoder."""

    def __init__(self, num_epochs, learning_rate, encoder_num: str, encoder_module: Optional[int] = -1,
                 encoder_layer: Optional[int] = -1):

        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.encoder_num = encoder_num

        # 0-based index. (0 is first module)
        # self.encoder_module = encoder_module  # Train classifier on output of this module (default: -1, last module)
        self._encoder_module = encoder_module
        self._encoder_layer = encoder_layer

    @property
    def encoder_module(self):
        return self._encoder_module

    @encoder_module.setter
    def encoder_module(self, value):
        if value < -1:
            raise ValueError(f"encoder_module must be -1 or greater. Got {value}")
        if value >= 3:
            raise ValueError(f"encoder_module must be less than 3. Got {value}")
        self._encoder_module = value

    @property
    def encoder_layer(self):
        return self._encoder_layer

    @encoder_layer.setter
    def encoder_layer(self, value):
        if value < -1:
            raise ValueError(f"encoder_layer must be -1 or greater. Got {value}")
        if value >= 8:
            raise ValueError(f"encoder_layer must be less than 8. Got {value}")
        self._encoder_layer = value


class ClassifierConfig(PostHocModel):
    def __init__(self, num_epochs, learning_rate, encoder_num: str,
                 bias: Optional[bool] = True, encoder_module: Optional[int] = -1, encoder_layer: Optional[int] = -1,
                 use_single_frame: Optional[bool] = False):
        super().__init__(num_epochs, learning_rate, encoder_num, encoder_module, encoder_layer)
        self.bias = bias
        self.use_single_frame = use_single_frame

    # to string
    def __str__(self):
        return f"ClassifierConfig(num_epochs={self.num_epochs}, learning_rate={self.learning_rate}, " \
               f"encoder_num={self.encoder_num}, bias={self.bias}, encoder_module={self.encoder_module}, " \
               f"encoder_layer={self.encoder_layer})"


class DecoderLoss(Enum):
    MSE = 0
    SPECTRAL = 1
    MSE_SPECTRAL = 2
    FFT = 3
    MSE_FFT = 4
    MEL = 5
    MSE_MEL = 6


class DecoderConfig(PostHocModel):
    def __init__(self, num_epochs, learning_rate, encoder_num: str,
                 architectures: Union[List[DecoderArchitectureConfig], List[VisionDecoderArchitectureConfig]],
                 decoder_loss: DecoderLoss,
                 encoder_module: Optional[int] = -1, encoder_layer: Optional[int] = -1):
        super().__init__(num_epochs, learning_rate, encoder_num, encoder_module, encoder_layer)

        self.architectures: Union[
            List[DecoderArchitectureConfig], List[VisionDecoderArchitectureConfig]] = architectures
        self.decoder_loss: DecoderLoss = decoder_loss

    def retrieve_correct_decoder_architecture(self) -> DecoderArchitectureConfig:
        # There are 3 architectures, one of each cnn module. (However, CPC works with single module,
        # so architectures match to certain layers of the module)
        module_idx = self.encoder_module
        layer_idx = self.encoder_layer

        if layer_idx == -1:  # final layer of specified module
            return self.architectures[module_idx]
        else:  # specific layer of specified module
            # [2, 5, 7] These are the layers in CPC_extended that correspond to the final layer of each module
            if layer_idx == 2:
                return self.architectures[0]  # equiv to module 0 in SIM/GIM
            elif layer_idx == 5:
                return self.architectures[1]
            elif layer_idx == 7:
                return self.architectures[2]  # equiv to final cnn module in SIM/GIM
            else:
                raise ValueError(f"A decoder architecture for module {module_idx} and layer {layer_idx} does not exist")


class OptionsConfig:
    def __init__(self, config_file, seed, validate, loss: Loss, encoder_config, experiment,
                 save_dir,
                 wandb_entity: str,
                 log_every_x_epochs,
                 classifier_configs: Dict[ClassifierKey, ClassifierConfig],
                 decoder_config: Optional[DecoderConfig],
                 vision_classifier_config: Optional[ClassifierConfig],

                 encoder_dataset: DataSetConfig,
                 post_hoc_dataset: DataSetConfig,

                 wandb_project_name: Optional[str] = "",
                 # two params used for local development. Not used in the cluster
                 use_wandb: Optional[bool] = True,
                 train: Optional[bool] = True,
                 compile_model: Optional[bool] = False,
                 profile: Optional[bool] = False,
                 fast_dev_run: Optional[bool] = False,
                 overfit_batches: Optional[int] = 0):
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

        self.decoder_config: Optional[DecoderConfig] = decoder_config

        self.vision_classifier_config: Optional[ClassifierConfig] = vision_classifier_config
        self.use_wandb = use_wandb
        self.train = train
        self.wandb_project_name = wandb_project_name
        self.wandb_entity = wandb_entity
        self.compile_model = compile_model
        self.profile = profile
        self._classifier_configs: Dict[ClassifierKey, ClassifierConfig] = classifier_configs

        self.encoder_dataset = encoder_dataset
        self.post_hoc_dataset = post_hoc_dataset
        self.fast_dev_run = fast_dev_run
        self.overfit_batches = overfit_batches

    @property
    def classifier_config(self) -> ClassifierConfig:
        classif, _ = self._get_classifier_and_key(self.post_hoc_dataset)
        return classif

    def _get_classifier_and_key(self, config: DataSetConfig) -> Tuple[ClassifierConfig, ClassifierKey]:
        if config.dataset == Dataset.LIBRISPEECH:
            if config.labels == Label.PHONES:
                return self._classifier_configs[ClassifierKey.LIBRI_PHONES], ClassifierKey.LIBRI_PHONES
            elif config.labels == Label.SPEAKERS:
                return self._classifier_configs[ClassifierKey.LIBRI_SPEAKERS], ClassifierKey.LIBRI_SPEAKERS
            else:
                raise ValueError(f"Invalid label {config.labels} for dataset {config.dataset}")
        elif config.dataset == Dataset.DE_BOER:
            if config.labels == Label.SYLLABLES:
                return self._classifier_configs[ClassifierKey.DE_BOER_SYLLABLES], ClassifierKey.DE_BOER_SYLLABLES
            elif config.labels == Label.VOWELS:
                return self._classifier_configs[ClassifierKey.DE_BOER_VOWELS], ClassifierKey.DE_BOER_SYLLABLES
            else:
                raise ValueError(f"Invalid label {config.labels} for dataset {config.dataset}")
        elif config.dataset == Dataset.RADIO:
            return self._classifier_configs[ClassifierKey.RADIO], ClassifierKey.RADIO
        else:
            raise ValueError(f"Invalid dataset {config.dataset}. (Vision datasets are not supported anymore)")

    def get_classifier_key(self):
        _, key = self._get_classifier_and_key(self.post_hoc_dataset)
        return key

    def __str__(self):
        return f"OptionsConfig(config_file={self.config_file}, seed={self.seed}, validate={self.validate}, " \
               f"loss={self.loss}, encoder_config={self.encoder_config}, experiment={self.experiment}, " \
               f"save_dir={self.save_dir}, wandb_entity={self.wandb_entity}, log_every_x_epochs={self.log_every_x_epochs}, " \
               f"classifier_configs={self._classifier_configs}, decoder_config={self.decoder_config}, " \
               f"vision_classifier_config={self.vision_classifier_config}, wandb_project_name={self.wandb_project_name}, " \
               f"use_wandb={self.use_wandb}, train={self.train}, compile_model={self.compile_model}, profile={self.profile})"
