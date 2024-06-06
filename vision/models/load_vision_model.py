from typing import Optional, Union

import torch

from config_code.config_classes import OptionsConfig, ClassifierConfig, ModelType, Dataset, DecoderConfig
from vision.models import FullModel, ClassificationModel
from utils import model_utils


def load_model_and_optimizer(opt: OptionsConfig,
                             downstream_config: Union[Optional[ClassifierConfig], Optional[DecoderConfig]],
                             num_GPU=None,
                             reload_model=False,
                             calc_loss=True) -> (FullModel.FullVisionModel, torch.optim.Optimizer):
    model: FullModel.FullVisionModel = FullModel.FullVisionModel(
        opt, calc_loss
    )

    lr = opt.encoder_config.learning_rate \
        if opt.model_type == ModelType.ONLY_ENCODER \
        else downstream_config.learning_rate

    # TODO, disabled by me
    # if opt.train_module != opt.model_splits and opt.model_splits > 1:  # Only train a part of the model
    #     optimizer = torch.optim.Adam(model.encoder[opt.train_module].parameters(), lr=lr)
    # else:  # Train the whole model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model, num_GPU = model_utils.distribute_over_GPUs(opt, model, num_GPU=num_GPU)

    if opt.model_type == ModelType.ONLY_ENCODER:
        model, optimizer = model_utils.reload_weights_for_training_encoder_vision_experiment(
            opt, model, optimizer, reload_model)
    else:  # ModelType.ONLY_DOWNSTREAM_TASK or ModelType.FULLY_SUPERVISED
        if type(downstream_config) == DecoderConfig:  # Decoder
            model, optimizer = model_utils.reload_weights_for_training_decoder_vision_experiment(
                opt, model, optimizer, reload_model, downstream_config)
        elif type(downstream_config) == ClassifierConfig:  # Classifier
            model, optimizer = model_utils.reload_weights_for_training_classifier_vision_experiment(
                opt, model, optimizer, reload_model, downstream_config)
        else:
            raise Exception(f"Invalid downstream config type: {type(downstream_config)}")

    return model, optimizer


def load_classification_model(opt: OptionsConfig) -> ClassificationModel.ClassificationModel:
    if opt.encoder_config.architecture.resnet_type == 34:
        in_channels = 256
    else:  # 50
        in_channels = 1024

    if opt.vision_classifier_config.dataset.dataset == Dataset.STL10:
        num_classes = 10
    elif opt.vision_classifier_config.dataset.dataset == Dataset.ANIMAL_WITH_ATTRIBUTES:
        num_classes = 50
    else:
        raise Exception("Invalid option")

    classification_model = ClassificationModel.ClassificationModel(
        classifier_config=opt.vision_classifier_config,
        in_channels=in_channels, num_classes=num_classes,
    ).to(opt.device)

    return classification_model
