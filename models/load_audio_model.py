import torch

from typing import Optional
from config_code.config_classes import Loss, ModelType, OptionsConfig, ClassifierConfig
from models import full_model
from models.full_model import FullModel
from utils import model_utils


def load_model_and_optimizer(
        opt: OptionsConfig, classifier_config: Optional[ClassifierConfig], reload_model=False, calc_accuracy=False,
        num_GPU=None) -> (FullModel, torch.optim.Optimizer):
    lr = opt.encoder_config.learning_rate
    # Initialize model.
    model: FullModel = full_model.FullModel(
        opt,
        calc_accuracy=calc_accuracy,
    )

    # Run on only one GPU for supervised losses.
    if opt.loss in [Loss.SUPERVISED_PHONES, Loss.SUPERVISED_SPEAKER]:
        num_GPU = 1

    model, num_GPU = model_utils.distribute_over_GPUs(
        opt, model, num_GPU=num_GPU)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if opt.model_type == ModelType.ONLY_ENCODER:
        model, optimizer = model_utils.reload_weights_for_training_encoder(opt, model, optimizer, reload_model)
    else:  # ModelType.ONLY_DOWNSTREAM_TASK or ModelType.FULLY_SUPERVISED
        model, optimizer = model_utils.reload_weights_for_training_classifier(
            opt, model, optimizer, reload_model, classifier_config)

    model.train()
    print(model)

    return model, optimizer
