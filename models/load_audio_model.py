import torch

from typing import Optional, Union
from config_code.config_classes import Loss, ModelType, OptionsConfig, ClassifierConfig, DecoderConfig
from decoder.decoderr import Decoder
from decoder.lit_decoder import LitDecoder
from models import full_model
from models.full_model import FullModel
from models.loss_supervised_syllables import Syllables_Loss
from utils import model_utils
import os


def load_model_and_optimizer(
        opt: OptionsConfig, classifier_config: Union[Optional[ClassifierConfig], Optional[DecoderConfig]],
        reload_model=False, calc_accuracy=False,
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


def load_decoder(opt: OptionsConfig, decoder: Decoder) -> Decoder:
    print(f"Loading decoder trained w/ loss: {opt.decoder_config.decoder_loss}")
    model_path = os.path.join(f"{opt.log_path}/model_0.ckpt")
    # Load the state dictionary
    state_dict = torch.load(model_path)

    # Load the state dictionary into the model
    decoder.load_state_dict(state_dict)
    return decoder


def load_classifier(opt: OptionsConfig, classifier: Syllables_Loss) -> Syllables_Loss:
    print(f"Loading classifier")
    model_path = os.path.join(f"{opt.log_path}/model_0.ckpt")
    # Load the state dictionary
    state_dict = torch.load(model_path)

    # Load the state dictionary into the model
    classifier.load_state_dict(state_dict)
    return classifier
