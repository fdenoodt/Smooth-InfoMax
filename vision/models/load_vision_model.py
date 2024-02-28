from typing import Optional

import torch

from config_code.config_classes import OptionsConfig, ClassifierConfig, ModelType
from vision.models import FullModel, ClassificationModel
from utils import model_utils


def load_model_and_optimizer(opt: OptionsConfig, classifier_config: Optional[ClassifierConfig], num_GPU=None,
                             reload_model=False, calc_loss=True):
    model = FullModel.FullVisionModel(
        opt, calc_loss
    )

    lr = opt.encoder_config.learning_rate \
        if opt.model_type == ModelType.ONLY_ENCODER \
        else classifier_config.learning_rate

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
        model, optimizer = model_utils.reload_weights_for_training_classifier_vision_experiment(
            opt, model, optimizer, reload_model, classifier_config)

    return model, optimizer


def load_classification_model(opt: OptionsConfig):
    # if opt.resnet == 34: # TODO: add resnet 34
    #     in_channels = 256
    # else:
    in_channels = 1024

    # if opt.dataset == "stl10":
    num_classes = 10
    # else:
    #     raise Exception("Invalid option")

    classification_model = ClassificationModel.ClassificationModel(
        in_channels=in_channels, num_classes=num_classes,
    ).to(opt.device)

    return classification_model
