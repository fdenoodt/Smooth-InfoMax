import torch

from config_code.config_classes import OptionsConfig
from vision.models import FullModel, ClassificationModel
from utils import model_utils


def load_model_and_optimizer(opt: OptionsConfig, num_GPU=None, reload_model=False, calc_loss=True):
    model = FullModel.FullVisionModel(
        opt, calc_loss
    )
    
    lr = opt.encoder_config.learning_rate # TODO: UNSRE IF THIS IS CORRECT

    if opt.train_module != opt.model_splits and opt.model_splits > 1:  # Only train a part of the model
        optimizer = torch.optim.Adam(model.encoder[opt.train_module].parameters(), lr=lr)
    else:  # Train the whole model
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model, num_GPU = model_utils.distribute_over_GPUs(opt, model, num_GPU=num_GPU)

    model, optimizer = model_utils.reload_weights_vision_experiment(
        opt, model, optimizer, reload_model=reload_model
    )


    return model, optimizer


def load_classification_model(opt):
    if opt.resnet == 34:
        in_channels = 256
    else:
        in_channels = 1024

    if opt.dataset == "stl10":
        num_classes = 10
    else:
        raise Exception("Invalid option")

    classification_model = ClassificationModel.ClassificationModel(
        in_channels=in_channels, num_classes=num_classes,
    ).to(opt.device)

    return classification_model
