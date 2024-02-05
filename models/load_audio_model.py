import torch

from configs.config_classes import Loss
from models import full_model
from models.full_model import FullModel
from utils import model_utils


def load_model_and_optimizer(
        opt, reload_model=False, calc_accuracy=False, num_GPU=None
) -> (FullModel, torch.optim.Optimizer):
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

    model, optimizer = model_utils.reload_weights_for_training_classifier(
        opt, model, optimizer, reload_model)

    model.train()
    print(model)

    return model, optimizer
